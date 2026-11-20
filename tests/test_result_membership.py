"""Nested preview counts and complete membership survive graph query/binder boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aag.engine.react_trace import collection_metadata


def graph_result(count=17):
    return {"results": [{"nodes": [{"node_key": f"entity_{index}"} for index in range(count)],
                         "relationships": [{"id": index, "source": "entity_0", "target": f"entity_{index % count}",
                                            "properties": {"quantity": index}} for index in range(31)]}], "count": 1}


def test_one_result_wrapper_preserves_nested_membership_counts_in_all_model_views():
    from aag.engine.react_executor import ReActExecutor, RunState

    value = graph_result()
    runner = ReActExecutor(reasoner=None, computing_engine=SimpleNamespace())
    bounded = runner._bounded_result({"success": True, "result": value})
    assert bounded["result_total_items"] == 1
    assert bounded["nested_collection_sizes"]["results[0].nodes"] == 17
    assert bounded["nested_collection_sizes"]["results[0].relationships"] == 31
    assert bounded["preview_collection_sizes"]["results[0].nodes"] == 8
    state = RunState(goal="Analyze the complete group")
    aid = runner._add_artifact(state, {"name": "subgraph"}, {"result": value}, {}, 1)
    assert state.prompt_view()["artifacts"][aid]["nested_collection_sizes"]["results[0].nodes"] == 17


def test_copying_unprovided_preview_ids_cannot_create_a_new_query_group():
    from aag.engine.react_executor import ReActExecutor, RunState

    engine = SimpleNamespace(execute_graph_query=MagicMock())
    runner = ReActExecutor(reasoner=None, computing_engine=engine)
    state = RunState(goal="Find the cohesive group containing the target")
    runner._add_artifact(state, {"name": "subgraph"}, {"result": graph_result()}, {}, 1)
    action = {"type": "graph_query", "name": "subgraph_by_nodes", "parameters": {
        "label": "Entity", "key": "node_key", "values": [f"entity_{index}" for index in range(8)]}}
    with pytest.raises(ValueError, match="complete upstream node selection"):
        asyncio.run(runner._execute_action(action, state))
    engine.execute_graph_query.assert_not_called()


def test_query_binder_extracts_every_member_from_the_full_source():
    from aag.engine.dependency_resolver import DataDependencyResolver
    from aag.engine.react_executor import ReActExecutor, RunState

    reasoner = SimpleNamespace(map_parameters=MagicMock(return_value={"mapping": {"values": {
        "parent_step_id": 1, "from_field": "results",
        "extract_code": "param_value = [node['node_key'] for node in value[0]['nodes']]"}}}))
    resolver = DataDependencyResolver(reasoner, None)
    engine = SimpleNamespace(execute_graph_query=MagicMock(return_value={"success": True, "results": [], "count": 0}))
    runner = ReActExecutor(reasoner=reasoner, computing_engine=engine, dependency_resolver=resolver)
    state = RunState(goal="Use the complete upstream node group")
    source = {"result": graph_result(), "query_scope": {"query_type": "subgraph", "parameters": {"hops": 2}}}
    aid = runner._add_artifact(state, {"name": "subgraph"}, source, {}, 1)
    action = {"type": "graph_query", "name": "subgraph_by_nodes", "parameters": {
        "label": "Entity", "key": "node_key", "values": []}, "consumed": [
            {"artifact_id": aid, "field": "results", "alias": "values", "use_as": "parameter"}]}
    result, _ = asyncio.run(runner._execute_action(action, state))
    values = engine.execute_graph_query.call_args.kwargs["parameters"]["values"]
    assert values == [f"entity_{index}" for index in range(17)]
    dependency = reasoner.map_parameters.call_args.kwargs["dependency_items"][0]
    assert len(dependency["sample_value"][0]["nodes"]) == 8
    assert dependency["nested_collection_sizes"]["[0].nodes"] == 17
    assert result["query_scope"]["parameters"]["values"] == values
    derived = runner._derived_evidence_scope(action, state)
    assert derived["query_inputs"][aid]["stored_collections"]["results[0].nodes"] == 17


def test_subgraph_projection_keeps_edge_identity_and_explicit_query_cap_metadata():
    from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient
    from aag.computing_engine.graph_query.parameters import normalize_query_parameters

    client = Neo4jGraphClient.__new__(Neo4jGraphClient)
    client.run = MagicMock(return_value=[{"nodes": [], "relationships": [], "node_count": 0,
                                        "relationship_count": 0, "path_limit_reached": True, "scope_is_complete": False}])
    value = client.subgraph_extract(("Entity", "node_key", "center"), limit_paths=1)
    query, parameters = client.run.call_args.args
    assert "properties(rel)" in query and "id(rel)" in query
    assert parameters["probe_limit"] == 2
    assert query.index("LIMIT $probe_limit") < query.index("collect(p)")
    assert value["scope_is_complete"] is False
    with pytest.raises(ValueError, match="Unsupported subgraph_by_nodes parameters"):
        normalize_query_parameters("subgraph_by_nodes", {"label": "Entity", "key": "node_key", "values": ["a"], "limit": 100}, None)


def test_nested_counts_do_not_expose_additional_row_values():
    metadata = collection_metadata(graph_result())
    assert "entity_16" not in str(metadata)
    assert metadata["preview_is_truncated"] is True
