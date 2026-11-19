"""Real NetworkX outputs must remain usable after passing through MCP JSON."""

import json

import networkx as nx

from aag.computing_engine.networkx_server import dynamic_tool_registry as registry


def test_real_community_output_is_a_structured_partition_instead_of_a_python_string():
    graph = nx.karate_club_graph()
    partition = nx.community.louvain_communities(graph, seed=0)
    result = json.loads(json.dumps(registry._serialize_result(partition)))
    assert isinstance(result, list) and all(isinstance(group, list) for group in result)
    assert {frozenset(group) for group in result} == {frozenset(group) for group in partition}
    assert set().union(*map(set, result)) == set(graph)


def test_real_cycle_iterator_is_evaluated_and_preserves_cycle_membership():
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    result = registry._serialize_result(nx.simple_cycles(graph, length_bound=3))
    assert isinstance(result, list) and len(result) == 1 and set(result[0]) == set(graph)


def test_capped_enumeration_is_explicitly_inexact(monkeypatch):
    monkeypatch.setattr(registry, "MAX_ITERATOR_ITEMS", 2)
    graph = nx.complete_graph(4, create_using=nx.DiGraph)
    result = registry._serialize_result(nx.simple_cycles(graph, length_bound=3))
    assert result["truncated"] is True and result["count_is_exact"] is False
    assert result["returned_items"] == len(result["items"]) == 2


def test_actual_pagerank_weight_contract_preserves_the_requested_edge_attribute():
    from aag.computing_engine.networkx_server.parameter_utils import normalize_parameters

    schema = registry.generate_input_schema(nx.pagerank)
    arguments = normalize_parameters("run_pagerank", {"weight": "strength"}, {"input_schema": schema})
    assert arguments == {"weight": "strength"}
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 1), ("b", "a", 10), ("b", "c", 1)], weight="strength")
    assert nx.pagerank(graph, **arguments) != nx.pagerank(graph, weight=None)
