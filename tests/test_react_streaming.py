"""Exercise ReAct streaming without a model server or external database."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("llama_index.core")

from aag.engine.react_executor import ReActExecutor
from aag.engine.scheduler import Scheduler
from aag.engine.router import QueryType
from aag.api.services.chat_service import ChatService


class Reasoner:
    def __init__(self):
        self.calls = 0

    def react_decide(self, **kwargs):
        self.calls += 1
        return {
            "thought": "Use the observed result." if self.calls > 1 else "Compute importance.",
            "checklist": [] if self.calls > 1 else ["Find important accounts"],
            "action": {"type": "finish" if self.calls > 1 else "algorithm", "name": "pagerank",
                       "parameters": {"alpha": 0.85}, "consumed": []},
        }

    def react_observe(self, **kwargs):
        result = kwargs["tool_result"]
        return {"summary": "Account A has score 0.7." if result["success"] else "The tool failed.",
                "evidence": [], "error": result.get("error")}

    def react_write_report(self, **kwargs):
        return "Account A is the most important. [a1]"


def executor(fails=False):
    engine = SimpleNamespace(
        get_algorithm_description=AsyncMock(return_value=("PageRank", {"name": "pagerank"})),
        run_algorithm=AsyncMock(return_value={"success": not fails, "result": {"A": 0.7},
                                              **({"error": "Tool unavailable"} if fails else {})}),
    )
    events = []
    return ReActExecutor(reasoner=Reasoner(), computing_engine=engine, callback=events.append), events


def test_react_stream_describes_actual_stages_and_returns_record():
    runner, events = executor()
    result = asyncio.run(runner.run("Find important accounts"))
    stages = [event["content"]["stage"] for event in events if event.get("contentType") == "react_status"]
    assert stages == ["deciding", "binding", "executing", "observing", "deciding", "reporting", "complete"]
    action = next(event for event in events if event.get("contentType") == "react_action")
    assert action["thought"] == "Compute importance."
    assert action["checklist"] == ["Find important accounts"]
    assert any(event.get("contentType") == "react_artifact" for event in events)
    assert result["dag_info"]["execution_driver"] == "react"
    assert result["analysis_result"].endswith("[a1]")


def test_failed_action_is_visible_as_failed_in_record():
    runner, events = executor(fails=True)
    result = asyncio.run(runner.run("Find important accounts"))
    observation = next(event["content"] for event in events if event.get("contentType") == "react_observation")
    assert observation["success"] is False
    assert "artifact_id" not in observation
    records = result["dag_info"]["steps"].values()
    assert any(step["record_type"] == "analysis" and step["status"] == "failed" for step in records)


@pytest.mark.parametrize("query_type", [QueryType.GRAPH, QueryType.GRAPH_QUERY])
def test_graph_queries_share_the_react_execution_entry(query_type):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.router = SimpleNamespace(route=lambda **_: SimpleNamespace(query_type=query_type, reason="test"))
    scheduler.current_dataset = [object()]
    scheduler.current_dataset_name = "AMLSim1K"
    scheduler.dataset_manager = SimpleNamespace(get_dataset_original_type=lambda _: "graph")
    scheduler._execute_graph = AsyncMock(return_value="ReAct result")
    assert asyncio.run(scheduler.execute("Question")) == "ReAct result"
    scheduler._execute_graph.assert_awaited_once_with("Question", decompose=True, mode="normal", callback=None)


@pytest.mark.parametrize("mode", ["normal", "react", "interact", "expert"])
def test_all_mode_aliases_execute_react_without_dag_planning(mode):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.computing_engine = SimpleNamespace(_initialized=True)
    scheduler.current_graph_dataset = object()
    scheduler.global_graph = object()
    scheduler._execute_graph_react = AsyncMock(return_value="ReAct result")
    result = asyncio.run(scheduler._execute_graph("Question", mode=mode))
    assert result == "ReAct result"
    scheduler._execute_graph_react.assert_awaited_once_with("Question", mode="normal", callback=None)
    with pytest.raises(RuntimeError, match="Only ReAct"):
        asyncio.run(scheduler.expert_start_analysis())


@pytest.mark.parametrize("mode", ["normal", "interact", "expert"])
def test_report_markdown_is_streamed_intact_for_old_and_new_clients(mode):
    report = "| Account | Score |\n| --- | --- |\n| A | 0.7 |\n\nRetried after Error: unavailable.\n\n```python\nprint('A')\n```"
    engine = SimpleNamespace(run=AsyncMock(return_value={
        "analysis_result": report,
        "dag_info": {"execution_driver": "react", "steps": {}, "edges": []},
    }))
    service = ChatService.__new__(ChatService)
    service.engine_service = SimpleNamespace(get_engine=lambda: engine)
    events = []
    asyncio.run(service.process_streaming_chat("Question", mode=mode, callback=events.append))
    reports = [event["content"] for event in events
               if event.get("type") == "result" and event.get("contentType") == "text"]
    assert reports == [report]
    assert engine.run.await_args.kwargs["mode"] == "normal"
    assert events[-1] == {"type": "stream_end"}


@pytest.fixture
def stream_app(monkeypatch):
    from flask import Flask
    from web.frontend.route import routes_chat_stream as routes

    app = Flask(__name__)
    app.register_blueprint(routes.bp)
    monkeypatch.setattr(routes, "get_background_loop", lambda: SimpleNamespace(
        call_soon_threadsafe=lambda fn: fn()))
    return app.test_client(), routes


def test_retired_dag_controls_do_not_execute_a_record(stream_app):
    client, _ = stream_app
    response = client.post("/api/chat/stream", json={
        "request_id": "old-confirm", "dataset": "AMLSim1K", "dag_confirm": "yes",
    })
    assert response.status_code == 400
    assert "ReAct" in response.json["error"]


def test_finished_react_stream_can_be_resumed(stream_app):
    client, routes = stream_app
    stream = routes._register_stream("react-resume-test")
    send = routes._make_sender(stream)
    send({"type": "result", "contentType": "react_status", "content": {"stage": "complete"}})
    send({"type": "stream_end"})
    response = client.post("/api/chat/stream", json={"request_id": "react-resume-test", "resume": True})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.decode().splitlines()]
    assert [event["seq"] for event in events] == [0, 1]
    assert events[0]["contentType"] == "react_status"
    assert events[-1]["type"] == "stream_end"


def test_saved_record_graph_can_be_refreshed_without_rerunning_analysis(stream_app, monkeypatch):
    client, routes = stream_app
    runner, _ = executor()
    result = asyncio.run(runner.run("Find important accounts"))
    calls = runner.reasoner.calls
    monkeypatch.setattr(routes.ReactTraceStore, "get", lambda self, run_id: {"events": result["react_trace"]})
    response = client.get("/api/chat/runs/" + "a" * 32 + "/dag")
    assert response.status_code == 200
    graph = response.json["data"]
    assert [node["record_type"] for node in graph["nodes"]] == ["analysis", "report"]
    assert runner.reasoner.calls == calls
