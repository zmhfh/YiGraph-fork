"""Recorded task graphs preserve evidence without exposing every lifecycle event."""

from copy import deepcopy

from aag.models.react_dag_projection import project_react_trace


def round_events(number, objective, artifact=None, success=True, consumed=None):
    action = {"type": "numeric_analysis", "name": "numeric_analysis", "objective": objective,
              "parameters": {"value": number}, "consumed": consumed or []}
    return [
        {"event_id": f"d{number}", "kind": "decision", "data": {"step": number, "thought": "Use evidence", "action": action}},
        {"event_id": f"t{number}", "kind": "tool_call", "data": {"step": number, "action": action, "parameters": action["parameters"]}},
        {"event_id": f"o{number}", "kind": "observation", "data": {
            "step": number, "action": action, "success": success, "summary": "Computed result" if success else "Unknown property",
            **({"artifact_id": artifact} if artifact else {}), **({"error": "Unknown property"} if not success else {})}},
    ]


def finish(status="completed"):
    return [
        {"kind": "decision", "data": {"step": 9, "action": {"type": "finish", "objective": "Summarize the analysis"}}},
        {"kind": "completion_review", "data": {"step": 9, "complete": status == "completed"}},
        {"kind": "report_review", "data": {"step": 9, "valid": True}},
        {"kind": "finish", "data": {"step": 9, "status": status, "termination_reason": status, "report": "Actual report"}},
    ]


def test_lifecycle_events_form_one_task_and_one_report_without_mutating_the_trace():
    trace = [{"kind": "start", "data": {"goal": "Analyze a graph"}}] + round_events(1, "Compute the ranking", "a1") + finish()
    original = deepcopy(trace)
    graph = project_react_trace(trace)
    assert len(graph["nodes"]) == 2
    analysis, report = graph["nodes"]
    assert analysis["question"] == "Compute the ranking" and analysis["record_type"] == "analysis"
    assert [event["kind"] for event in analysis["record"]["attempts"][0]["events"]] == ["decision", "tool_call", "observation"]
    assert analysis["record"]["artifact_ids"] == ["a1"]
    assert report["record_type"] == "report" and report["status"] == "success"
    assert graph["edges"] == [{"from": "1", "to": "2", "kind": "summary"}]
    assert trace == original


def test_a_repaired_objective_keeps_its_failure_inside_one_task():
    trace = round_events(1, "Calculate a measure", success=False) + round_events(2, "Calculate a measure", "a2")
    graph = project_react_trace(trace)
    assert len(graph["nodes"]) == 1
    node = graph["nodes"][0]
    assert node["status"] == "success" and node["error"] is None
    assert node["record"]["rounds"] == [1, 2]
    assert node["record"]["attempts"][0]["events"][-1]["data"]["error"] == "Unknown property"


def test_dependency_branches_follow_artifacts_instead_of_linking_every_event_in_order():
    trace = round_events(1, "Select entities", "a1")
    trace += round_events(2, "Compute scores", "a2", consumed=["a1"])
    trace += round_events(3, "Compute a separate measure", "a3", consumed=[{"artifact_id": "a1", "field": "entities"}])
    graph = project_react_trace(trace + finish())
    assert graph["edges"] == [
        {"from": "1", "to": "2", "kind": "data"}, {"from": "1", "to": "3", "kind": "data"},
        {"from": "2", "to": "4", "kind": "summary"}, {"from": "3", "to": "4", "kind": "summary"}]


def test_execution_order_is_distinct_from_a_result_reference_and_failures_stay_failed():
    trace = round_events(1, "Find a node", "a1") + round_events(2, "Calculate a measure", success=False) + finish("incomplete")
    graph = project_react_trace(trace)
    assert graph["edges"][0]["kind"] == "sequence"
    assert graph["nodes"][1]["status"] == "failed" and graph["nodes"][1]["error"] == "Unknown property"
    assert graph["nodes"][-1]["status"] == "failed"


def test_frontend_projection_preserves_task_metadata_without_running_dag_validation(monkeypatch):
    from aag.models.graph_workflow_dag import GraphWorkflowDAG, dag_info_to_frontend_format

    def no_planning(*args):
        raise AssertionError("A record is not an execution plan")

    monkeypatch.setattr(GraphWorkflowDAG, "add_dependency", no_planning)
    monkeypatch.setattr(GraphWorkflowDAG, "topological_order", no_planning)
    dag = GraphWorkflowDAG.from_react_trace(round_events(1, "Compute the ranking", "a1") + finish())
    payload = dag_info_to_frontend_format(dag.get_dag_info())
    assert len(payload["nodes"]) == 2
    assert payload["nodes"][0]["algorithm"] == "numeric_analysis"
    assert payload["nodes"][0]["summary"] == "Computed result"
    assert payload["nodes"][0]["record"]["artifact_ids"] == ["a1"]
    assert payload["nodes"][1]["tasktype"] == "Analysis Report"
    assert payload["edges"][0]["kind"] == "summary"
