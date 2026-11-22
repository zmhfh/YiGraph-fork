"""Task extraction uses full outputs and preserves evidence through processing errors."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("llama_index.core")
from aag.computing_engine.code_executor import DynamicCodeExecutor
from aag.computing_engine.computing_engine import ComputingEngine
from aag.engine.react_executor import ReActExecutor, RunState
from aag.engine.react_trace import ReactTraceStore


CODE = """def process(data):
    ranked = sorted(data.items(), key=lambda item: item[1], reverse=True)
    return {"highest": {"node": ranked[0][0], "score": ranked[0][1]}, "top_nodes": ranked[:3]}
"""
SPEC = {"required": True, "code": CODE, "output_schema": {"fields": {
    "highest": {"type": "dict", "field_description": "Maximum over the complete node-score mapping."},
    "top_nodes": {"type": "list", "field_description": "Three highest-scoring nodes from the full mapping."}}}}


def action():
    return {"type": "algorithm", "name": "pagerank", "objective": "Find the globally highest score",
            "parameters": {"alpha": 0.85, "weight": None}, "consumed": []}


def make_runner(reasoner, raw, **kwargs):
    executor = ComputingEngine.__new__(ComputingEngine)
    executor.code_executor = DynamicCodeExecutor(auto_install=False)
    engine = SimpleNamespace(
        get_algorithm_description=AsyncMock(return_value=("Return a complete node-score dictionary", {"name": "pagerank"})),
        run_algorithm=AsyncMock(return_value={"success": True, "result": raw}),
        execute_code=executor.execute_code,
    )
    return ReActExecutor(reasoner=reasoner, computing_engine=engine, **kwargs), engine


def test_prepared_parameters_still_get_task_fields_from_scores_outside_the_preview():
    raw = {f"entity_{index}": index / 100 for index in range(40)}
    reasoner = SimpleNamespace(generate_algorithm_postprocessing=MagicMock(return_value=SPEC))
    runner, engine = make_runner(reasoner, raw)
    state = RunState(goal="Identify the highest-influence node")
    result, metadata = asyncio.run(runner._execute_action(action(), state))
    supplied = reasoner.generate_algorithm_postprocessing.call_args.kwargs
    assert supplied["parameters"] == {"alpha": 0.85, "weight": None}
    assert len(supplied["result_view"]["result"]) == 8
    assert result["result"]["highest"] == {"node": "entity_39", "score": 0.39}
    assert "complete" in runner._bounded_result(result)["output_schema"]["fields"]["highest"]["field_description"]
    assert result["result"]["original_result"] == raw and len(raw) == 40
    assert engine.run_algorithm.await_count == 1
    assert engine.run_algorithm.await_args.args[2] is None
    artifact_id = runner._add_artifact(state, action(), result, metadata, 1)
    preview = state.prompt_view()["artifacts"][artifact_id]
    assert preview["sample"]["highest"]["node"] == "entity_39"
    assert preview["collection_sizes"]["original_result"] == 40
    assert "complete" in preview["fields"]["highest"]["field_description"]


def test_processing_cannot_replace_or_mutate_the_preserved_algorithm_result():
    raw = {"first": 1, "second": 3}
    reasoner = SimpleNamespace(generate_algorithm_postprocessing=MagicMock(return_value={
        "required": True, "code": "def process(data):\n    data.pop('first')\n    return {'original_result': {}, 'count': len(data)}",
        "output_schema": {"fields": {"count": {"type": "int"}}}}))
    runner, _ = make_runner(reasoner, raw)
    result, _ = asyncio.run(runner._execute_action(action(), RunState(goal="Process data")))
    assert result["result"]["original_result"] == {"first": 1, "second": 3}
    assert raw == {"first": 1, "second": 3}


def test_failed_extraction_retains_raw_artifact_and_can_be_repaired_without_repeating_the_algorithm(tmp_path):
    decisions = iter([
        {"action": action(), "checklist": ["Find the highest score"]},
        {"action": {"type": "numeric_analysis", "name": "numeric_analysis", "parameters": {
            "code": "def process(scores):\n    key = max(scores, key=scores.get)\n    return {'highest': key, 'score': scores[key]}"},
            "consumed": [{"artifact_id": "a1", "field": "original_result", "alias": "scores"}]}, "checklist": []},
        {"action": {"type": "finish", "name": "finish"}, "checklist": []},
    ])
    reasoner = SimpleNamespace(
        react_decide=lambda **kwargs: next(decisions),
        react_observe=lambda **kwargs: {"summary": "Actual tool evidence"},
        react_review_completion=lambda **kwargs: {"complete": True, "missing_goals": []},
        react_write_report=lambda **kwargs: "Highest score computed from complete data [a2].",
        generate_algorithm_postprocessing=MagicMock(return_value={"required": True,
            "code": "def process(data):\n    return {'error': 'Bad extraction'}", "output_schema": {}}),
    )
    runner, engine = make_runner(reasoner, {"left": 1, "right": 4},
                                 config={"max_steps": 3, "max_retries_per_action": 0}, trace_store=ReactTraceStore(tmp_path))
    result = asyncio.run(runner.run("Find the highest score"))
    record = runner.trace_store.get(result["run_id"])
    assert result["analysis_status"] == "completed"
    assert record["artifacts"]["a1"]["value"]["original_result"] == {"left": 1, "right": 4}
    assert record["artifacts"]["a2"]["value"] == {"highest": "right", "score": 4}
    observations = [event["data"] for event in runner.events if event["kind"] == "observation"]
    assert observations[0]["success"] is False and observations[0]["artifact_id"] == "a1"
    assert "Bad extraction" in observations[0]["error"]
    assert engine.run_algorithm.await_count == 1


def test_failed_extraction_at_the_step_limit_is_incomplete_with_usable_raw_evidence():
    reasoner = SimpleNamespace(
        react_decide=lambda **kwargs: {"action": action()},
        react_observe=lambda **kwargs: {"summary": "Extraction failed"},
        react_write_report=lambda **kwargs: "The requested maximum remains uncomputed.",
        generate_algorithm_postprocessing=lambda **kwargs: {"required": True, "code": "raise ValueError('Broken code')"},
    )
    runner, _ = make_runner(reasoner, {"left": 1, "right": 4}, config={"max_steps": 1, "max_retries_per_action": 0})
    result = asyncio.run(runner.run("Find the highest score"))
    assert result["analysis_status"] == "incomplete"
    assert result["state"]["artifacts"]["a1"]["sample"]["original_result"] == {"left": 1, "right": 4}


def test_repair_attempts_do_not_reexecute_the_algorithm_and_partial_scope_survives():
    reasoner = SimpleNamespace(generate_algorithm_postprocessing=MagicMock(side_effect=[
        {"required": True, "code": "def process(data):\n    return {'error': 'Wrong shape'}"},
        {"required": True, "code": "def process(data):\n    return {'returned_count': len(data['items'])}",
         "output_schema": {"fields": {"returned_count": {"type": "int"}}}},
    ]))
    raw = {"items": [["a", "b"]], "truncated": True, "count_is_exact": False}
    runner, engine = make_runner(reasoner, raw)
    state = RunState(goal="Describe retained cycles")
    result, metadata = asyncio.run(runner._execute_action(action(), state))
    assert result["postprocessing"]["status"] == "success"
    assert len(reasoner.generate_algorithm_postprocessing.call_args.kwargs["error_history"]) == 1
    assert engine.run_algorithm.await_count == 1
    artifact = runner._add_artifact(state, action(), result, metadata, 1)
    assert state.artifacts[artifact]["evidence_scope"]["source_truncated"] is True
    assert state.artifacts[artifact]["value"]["original_result"]["count_is_exact"] is False


def test_scalar_results_skip_an_unnecessary_binder_call():
    reasoner = SimpleNamespace(generate_algorithm_postprocessing=MagicMock())
    runner, _ = make_runner(reasoner, 186)
    result, _ = asyncio.run(runner._execute_action(action(), RunState(goal="Count components")))
    assert result["result"] == 186
    reasoner.generate_algorithm_postprocessing.assert_not_called()


def test_actual_initialized_subgraph_metadata_survives_extraction_and_artifact_preview():
    from aag.config.data_upload_config import DatasetConfig
    from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
    from aag.expert_search_engine.database.datatype import GraphData, VertexData, EdgeData

    graph = GraphData([VertexData(name) for name in ("a", "b", "c")],
                      [EdgeData("a", "b", rank="first"), EdgeData("a", "b", rank="second"), EdgeData("b", "c", rank="third")])
    config = DatasetConfig.from_dict({"name": "Other", "type": "graph", "description": "Synthetic payments in one source-defined currency and clock window.", "schema": {
        "vertex": [], "edge": [], "graph_structure": {"directed": True, "multigraph": True}, "graph_store_info": {}}})
    reasoner = SimpleNamespace(generate_algorithm_postprocessing=MagicMock(return_value=SPEC))
    resolver = SimpleNamespace(prepare_react_inputs=AsyncMock(return_value={
        "parameters": {"alpha": 0.85}, "graph_adapter_result": {"nodes": ["a", "b"], "edges": [["a", "b"]],
            "edge_selection": "all_parallel"}, "errors": []}))
    runner, engine = make_runner(reasoner, {}, global_graph=graph, current_graph_dataset=config, dependency_resolver=resolver)
    processor = GraphComputationProcessor()
    reasoner.generate_algorithm_postprocessing.return_value = {"required": True, "code": '''def process(data):
    metadata = dict(execution_context["working_graph"]["metadata"])
    execution_context["working_graph"]["metadata"]["edge_count"] = -1
    return {"actual_graph": metadata, "weight_argument": execution_context["algorithm_parameters"].get("weight")}
''', "output_schema": {}}

    async def execute(name, parameters, *args, **kwargs):
        if name == "initialize_graph":
            processor.create_graph_from_edges(
                [VertexData.from_dict(item) for item in parameters["vertices"]],
                [EdgeData.from_dict(item) for item in parameters["edges"]], directed=True, multiedge=True)
            return {"success": True, "result": processor.get_graph_info()}
        return {"success": True, "result": processor.run_pagerank()}

    engine.run_algorithm.side_effect = execute
    request = {**action(), "consumed": [{"artifact_id": "a0", "use_as": "graph"}]}
    state = RunState(goal="Rank a selected subgraph, retaining parallel events")
    result, metadata = asyncio.run(runner._execute_action(request, state))
    actual = result["evidence_scope"]["working_graph"]["metadata"]
    assert {key: actual[key] for key in ("vertex_count", "edge_count", "directed", "multigraph")} == {
        "vertex_count": 2, "edge_count": 2, "directed": True, "multigraph": True}
    assert set(result["result"]["original_result"]) == {"a", "b"}
    assert result["result"]["actual_graph"] == actual
    assert actual["edge_count"] == 2  # Generated code cannot mutate authoritative scope metadata.
    supplied = reasoner.generate_algorithm_postprocessing.call_args.kwargs
    assert supplied["execution_scope"]["evidence_scope"]["working_graph"]["metadata"] == actual
    artifact_id = runner._add_artifact(state, request, result, metadata, 1)
    assert state.prompt_view()["artifacts"][artifact_id]["evidence_scope"]["working_graph"]["metadata"] == actual
    assert state.prompt_view()["artifacts"][artifact_id]["evidence_scope"]["working_graph"]["source_dataset_description"] == config.description


def test_an_extractor_generated_with_missing_parameters_is_reused():
    reasoner = SimpleNamespace(
        extract_parameters_with_postprocess_new=MagicMock(return_value={
            "parameters": {"alpha": 0.85}, "post_processing_code": SPEC}),
        generate_algorithm_postprocessing=MagicMock(),
    )
    runner, _ = make_runner(reasoner, {"first": 1, "last": 4})
    request = {**action(), "parameters": {}}
    result, metadata = asyncio.run(runner._execute_action(request, RunState(goal="Find the highest score")))
    assert result["result"]["highest"] == {"node": "last", "score": 4}
    assert "highest" in metadata["output_params"]["fields"]
    reasoner.generate_algorithm_postprocessing.assert_not_called()


def test_numeric_field_meaning_reaches_report_models_instead_of_only_its_python_type():
    reasoner = SimpleNamespace(react_write_report=MagicMock(return_value="Counted components [a1]."))
    runner, _ = make_runner(reasoner, {})
    state = RunState(goal="Count weak components", status="completed")
    result = {"success": True, "result": {"components": 3}, "output_schema": {"fields": {
        "components": {"type": "int", "field_description": "Number of weakly connected components, ignoring edge direction."}}}}
    runner._add_artifact(state, {"name": "numeric_analysis"}, result, {}, 1)
    asyncio.run(runner._write_report(state, state.goal))
    evidence = reasoner.react_write_report.call_args.kwargs["artifacts"]["a1"]
    assert "weakly connected" in evidence["fields"]["components"]["field_description"]
