"""Dataset-independent semantic bindings, including the actual ReAct tool route."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from aag.config.data_upload_config import DatasetConfig, load_data_upload_config, save_data_config
from aag.computing_engine.schema_roles import bind_temporal_path_parameters, temporal_path_schema


def dataset_entry():
    return {"name": "activity", "type": "graph", "description": "A different property vocabulary",
            "schema": {"vertex": [{"type": "entity", "path": "nodes.csv", "format": "csv", "query_field": "title",
                                    "id_field": "external_ref", "semantic_roles": {"screen_flag": "reviewed"}}],
                       "edge": [{"type": "movement", "path": "edges.csv", "format": "csv", "source_field": "origin",
                                 "target_field": "destination", "rank_field": "event_id",
                                 "semantic_roles": {"value": "quantity", "timestamp": "happened_at",
                                                    "timestamp_unit": "unix_ms", "screen_flag": "marked"}}],
                       "graph_structure": {}, "graph_store_info": {}}}


def test_yaml_load_save_preserves_semantic_roles_and_binds_the_new_vocabulary(tmp_path):
    entry = dataset_entry()
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump({"datasets": [entry]}))
    config = load_data_upload_config(str(path))
    save_data_config(config, str(path))
    restored = load_data_upload_config(str(path)).datasets[0]
    assert restored.schema.edge[0].rank_field == "event_id"
    assert restored.schema.edge[0].semantic_roles == entry["schema"]["edge"][0]["semantic_roles"]
    assert bind_temporal_path_parameters(restored, {"center": "hub"}) == {
        "center": "hub", "amount_field": "quantity", "timestamp_field": "happened_at",
        "timestamp_unit": "unix_ms", "node_id_field": "external_ref"}
    assert temporal_path_schema(restored)["declared_screening_fields"] == {
        "node_flag_field": "reviewed", "edge_flag_field": "marked"}


def test_undeclared_and_ambiguous_roles_require_explicit_bindings():
    entry = dataset_entry()
    entry["schema"]["edge"].append(deepcopy(entry["schema"]["edge"][0]))
    assert "amount_field" not in temporal_path_schema(entry)["defaults"]
    with pytest.raises(ValueError, match="Missing temporal-path field bindings"):
        bind_temporal_path_parameters(entry, {"center": "hub"})
    explicit = {"amount_field": "quantity", "timestamp_field": "happened_at", "timestamp_unit": "unix_ms"}
    assert bind_temporal_path_parameters(None, explicit) == explicit
    del entry["schema"]["edge"][1]
    entry["schema"]["edge"][0]["semantic_roles"] = {}
    with pytest.raises(ValueError, match="Missing temporal-path field bindings"):
        bind_temporal_path_parameters(entry, {"center": "hub"})


def test_model_overrides_declared_fields_without_mutating_configuration():
    entry = dataset_entry()
    bound = bind_temporal_path_parameters(entry, {"amount_field": "alternate", "node_id_field": None})
    assert bound["amount_field"] == "alternate" and bound["node_id_field"] is None
    assert temporal_path_schema(entry)["defaults"]["amount_field"] == "quantity"


@pytest.mark.parametrize("action_type", ["algorithm", "numeric_analysis"])
def test_react_executes_schema_bound_paths_on_a_different_dataset(action_type):
    pytest.importorskip("llama_index.core")
    from aag.engine.react_executor import ReActExecutor, RunState
    dataset = DatasetConfig.from_dict(dataset_entry())
    graph = SimpleNamespace(
        vertices=[SimpleNamespace(vid=name, properties={"external_ref": index, "reviewed": name == "middle"})
                  for index, name in enumerate(["upstream", "hub", "middle", "downstream"])],
        edges=[SimpleNamespace(src=source, dst=target,
                               properties={"quantity": value, "happened_at": moment, "marked": False})
               for source, target, value, moment in [("upstream", "hub", 100, 1000),
                                                     ("hub", "middle", 90, 2000), ("middle", "downstream", 80, 3000)]])
    runner = ReActExecutor(reasoner=None, computing_engine=SimpleNamespace(),
                           current_graph_dataset=dataset, global_graph=graph)
    tool = next(item for item in runner.available_tools() if item["id"] == "temporal_paths")
    assert tool["required_parameters"] == ["center"]
    assert tool["type"] == "algorithm" and tool["engine"] == "builtin"
    assert tool["dataset_bindings"]["defaults"]["amount_field"] == "quantity"
    action = {"type": action_type, "name": "temporal_paths", "parameters": {
        "center": 1, "require_path_flag": True, "node_flag_field": "reviewed"}}
    result, _ = asyncio.run(runner._execute_action(action, RunState(goal="Analyze scoped flow")))
    value = result["result"]
    assert value["path_count"] == 1 and value["qualified_center_outflow_amount"] == 90
    assert value["center"] == "hub"
    assert value["largest_total_amount_account"]["node_key"] == "hub"
    assert value["largest_total_amount_account"]["source_identifier"] == {"field": "external_ref", "value": 1}
    assert runner.events[-1]["data"]["parameters"]["timestamp_unit"] == "unix_ms"
    undeclared = ReActExecutor(reasoner=None, computing_engine=SimpleNamespace())
    tool = next(item for item in undeclared.available_tools() if item["id"] == "temporal_paths")
    assert tool["required_parameters"] == ["center", "amount_field", "timestamp_field"]
