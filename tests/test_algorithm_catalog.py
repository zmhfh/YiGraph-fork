"""The decision model receives executable contracts rather than knowledge-only names."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aag.computing_engine.algorithm_catalog import algorithm_tool_catalog


def test_runtime_descriptions_and_parameters_override_stale_knowledge():
    runtime = [{"id": "shortest_path", "engine": "networkx", "tool_name": "run_shortest_path",
                "description": "Compute a minimum-hop or minimum-weight path.\n\nParameters\n----------\nG : graph",
                "input_schema": {"parameters": {
                    "G": {"type": "graph", "required": True}, "source": {"type": "node", "required": True},
                    "target": {"type": "node", "default": None}, "weight": {"type": "str", "default": None}},
                    "required": ["G", "source"]}, "system_supplied_parameters": ["G"]}]
    knowledge = {"shortest_path": {"description": "Stale description", "task_type_id": "Path",
                                  "Deployment_method": {"input_schema": {"parameters": {"invented": {"required": True}}}}},
                 "not_deployed": {"Principles": {"description": "Not callable"}}}
    tools = algorithm_tool_catalog(runtime, knowledge)
    assert [tool["id"] for tool in tools] == ["shortest_path"]
    tool = tools[0]
    assert tool["description"] == "Compute a minimum-hop or minimum-weight path."
    assert tool["required_parameters"] == ["source"]
    assert set(tool["parameter_schema"]) == {"source", "target", "weight"}
    assert tool["parameter_schema"]["weight"]["default"] is None


def test_capabilities_require_a_mapping_a_connected_engine_and_an_actual_tool():
    from aag.computing_engine.computing_engine import ComputingEngine

    engine = ComputingEngine.__new__(ComputingEngine)
    engine.parameter_modules = {}
    engine.algorithm_tool_mapping = {
        "networkx": {"initialize_graph": "run_initialize_graph", "connected_components": "run_connected_components",
                     "clustering": "run_clustering", "missing": "run_missing"},
        "offline": {"unavailable": "run_unavailable"},
    }
    engine.clients = {"networkx": SimpleNamespace(available_tools={
        "run_initialize_graph": {"description": "setup"},
        "run_connected_components": {"description": "Connectivity components", "input_schema": {}},
        "run_clustering": {"description": "Clustering coefficient", "input_schema": {}},
        "run_unmapped": {"description": "Not configured in the executor"},
    })}
    tools = {tool["id"]: tool for tool in engine.algorithm_capabilities()}
    assert set(tools) == {"connected_components", "clustering"}
    assert "weakly_connected_components" in tools["connected_components"]["implementation_behavior"]["directed_input"]
    assert "undirected" in tools["clustering"]["implementation_behavior"]["directed_input"]


def test_react_initializes_the_engine_that_will_run_the_selected_algorithm():
    from aag.engine.react_executor import ReActExecutor
    from aag.expert_search_engine.database.datatype import GraphData

    engine = SimpleNamespace(
        algorithm_tool_mapping={"networkx": {"initialize_graph": "run_initialize_graph", "pagerank": "run_pagerank"},
                                "pyg": {"initialize_graph_pyg": "run_initialize_graph", "gcn_node_classification": "run_gcn_node_classification"}},
        run_algorithm=AsyncMock(return_value={"success": True}),
    )
    runner = ReActExecutor(reasoner=None, computing_engine=engine, global_graph=GraphData([], []))
    asyncio.run(runner._initialize_working_graph(None, "gcn_node_classification"))
    assert engine.run_algorithm.await_args.args[0] == "initialize_graph_pyg"
    asyncio.run(runner._initialize_working_graph(None, "pagerank"))
    assert engine.run_algorithm.await_args.args[0] == "initialize_graph"
