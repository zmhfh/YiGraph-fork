"""Regressions for premature finish, hidden tool failures and schema repair."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aag.computing_engine.graph_query.parameters import normalize_query_parameters
from aag.engine.react_trace import ReactTraceStore, preview
from aag.reasoner.prompt_template.react_prompts import (
    DECISION_INSTRUCTIONS, OBSERVER_INSTRUCTIONS, COMPLETION_INSTRUCTIONS, REPORT_INSTRUCTIONS,
    REPORT_REVIEW_INSTRUCTIONS,
    ALGORITHM_POSTPROCESSING_INSTRUCTIONS,
)

SCHEMA = {
    "node_labels": {"Account": {"properties": ["node_key", "acct_id"]}},
    "relationship_types": {"TRANSFER": {"properties": ["base_amt", "tran_timestamp", "is_sar"]}},
}


def test_anna_neighbor_arguments_are_repaired_without_losing_amounts():
    params = normalize_query_parameters("neighbor_query", {
        "label": "account", "key": "name", "value": "Anna Lee", "rel_type": "transfer",
        "return_fields": ["a.node_key", "b.node_key", "r.base_amt", "r.tran_timestamp"],
        "order_by": {"field": "r.base_amt", "direction": "DESC"},
    }, SCHEMA)
    assert params["label"] == "Account" and params["key"] == "node_key"
    assert params["rel_type"] == "TRANSFER"
    assert params["order_direction"] == "DESC"
    assert params["order_by"] == "rel_base_amt"
    assert params["return_fields"] == ["start.node_key", "nbr.node_key", "rel.base_amt", "rel.tran_timestamp"]


def test_bare_transaction_fields_use_relationship_scope():
    params = normalize_query_parameters("neighbor_query", {
        "label": "Account", "key": "node_key", "value": "Anna Lee",
        "return_fields": ["acct_id", "base_amt", "tran_timestamp"],
    }, SCHEMA)
    assert params["return_fields"] == ["nbr.acct_id", "rel.base_amt", "rel.tran_timestamp"]


def test_graph_filter_aliases_use_the_actual_template_variable():
    params = normalize_query_parameters("neighbor_query", {
        "label": "Account", "key": "node_key", "value": "Anna Lee", "where": "rel.is_sar = true",
        "return_fields": ["nbr_acct_id", "rel_base_amt"], "direction": "outgoing",
    }, SCHEMA)
    assert params["where"] == "r.is_sar = true"
    assert params["direction"] == "out"
    assert params["return_fields"] == ["nbr.acct_id", "rel.base_amt"]


def test_relationship_condition_list_is_converted_to_property_mapping():
    params = normalize_query_parameters("relationship_filter", {
        "rel_type": "TRANSFER", "rel_conditions": [{"field": "is_sar", "operator": "=", "value": True}],
    }, SCHEMA)
    assert params["rel_conditions"] == {"is_sar": ["=", True]}


def test_unknown_money_property_is_not_silently_returned_as_null():
    with pytest.raises(ValueError, match="base_amt"):
        normalize_query_parameters("neighbor_query", {
            "label": "Account", "key": "node_key", "value": "Anna Lee", "return_fields": ["nbr.amount"],
        }, SCHEMA)


def test_relationship_endpoint_aliases_match_the_actual_tool_contract():
    params = normalize_query_parameters("relationship_filter", {
        "rel_type": "TRANSFER", "return_fields": ["start.node_key", "end.acct_id", "r.base_amt"],
        "order_by": "rel.base_amt", "order_direction": "DESC",
    }, SCHEMA)
    assert params["return_fields"] == ["from.node_key", "to.acct_id", "rel.base_amt"]
    assert params["order_by"] == "base_amt"


def test_unsupported_endpoint_filter_is_not_silently_ignored():
    with pytest.raises(ValueError, match="neighbor_query"):
        normalize_query_parameters("relationship_filter", {
            "rel_type": "TRANSFER", "start_conditions": {"node_key": "Anna Lee"},
        }, SCHEMA)


def test_computed_path_amounts_require_an_explicit_actual_property():
    schema = {"node_labels": {"Entity": {"properties": ["node_key"]}},
              "relationship_types": {"MOVEMENT": {"properties": ["quantity", "happened_at"]}}}
    arguments = {"label": "Entity", "key": "node_key", "v1": "A", "v2": "B",
                 "rel_type": "MOVEMENT", "return_fields": ["nodes", "totalAmount"]}
    with pytest.raises(ValueError, match="require amount_field"):
        normalize_query_parameters("path_query", arguments, schema)
    with pytest.raises(ValueError, match="Unknown relationship amount property"):
        normalize_query_parameters("path_query", {**arguments, "amount_field": "base_amt"}, schema)
    assert normalize_query_parameters("path_query", {**arguments, "amount_field": "quantity"}, schema)["amount_field"] == "quantity"


def test_path_amount_projection_uses_the_bound_property_and_rejects_missing_mapping():
    from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient
    client = Neo4jGraphClient.__new__(Neo4jGraphClient)
    client.run = MagicMock(return_value=[])
    endpoints = (("Entity", "node_key", "A"), ("Entity", "node_key", "B"))
    with pytest.raises(ValueError, match="explicit amount_field"):
        client.paths_between(*endpoints, return_fields=["totalAmount"])
    client.paths_between(*endpoints, amount_field="quantity", return_fields=["totalAmount", "maxAmount", "minAmount", "avgAmount"])
    query = client.run.call_args[0][0]
    assert "r.`quantity`" in query and "base_amt" not in query
    client.paths_between(*endpoints, return_fields=["nodes", "relationships"])


def test_fixed_react_instructions_are_english():
    assert all(text.isascii() for text in [DECISION_INSTRUCTIONS, OBSERVER_INSTRUCTIONS,
                                           COMPLETION_INSTRUCTIONS, REPORT_INSTRUCTIONS, REPORT_REVIEW_INSTRUCTIONS,
                                           ALGORITHM_POSTPROCESSING_INSTRUCTIONS])


def test_preview_preserves_entity_identity_even_after_many_other_fields():
    value = {**{f"irrelevant_{index}": index for index in range(20)},
             "node_key": "Anna Lee", "acct_id": 33, "first_name": "Lee", "last_name": "Anna"}
    shown = preview(value)
    assert shown["node_key"] == "Anna Lee" and shown["acct_id"] == 33
    assert shown["first_name"] == "Lee"


@pytest.fixture
def react_classes():
    pytest.importorskip("llama_index.core")
    from aag.engine.react_executor import ReActExecutor
    return ReActExecutor


class ScriptedReasoner:
    def __init__(self, decisions, reviews=None):
        self.decisions = iter(decisions)
        self.reviews = iter(reviews or [])
        self.states = []

    def react_decide(self, **kwargs):
        self.states.append(kwargs["state"])
        return next(self.decisions)

    def react_observe(self, **kwargs):
        return {"summary": "Actual tool result", "evidence": []}

    def react_review_completion(self, **kwargs):
        return next(self.reviews)

    def react_write_report(self, **kwargs):
        return "Available computation evidence."


def decision(kind="algorithm", alpha=0.85):
    return {"thought": "Compute the next output", "checklist": [],
            "action": {"type": kind, "name": "pagerank", "parameters": {"alpha": alpha}, "consumed": []}}


def make_runner(cls, reasoner, **kwargs):
    engine = SimpleNamespace(
        get_algorithm_description=AsyncMock(return_value=("PageRank", {"name": "pagerank"})),
        run_algorithm=AsyncMock(return_value={"success": True, "result": {"score": 0.7}}),
    )
    events = []
    return cls(reasoner=reasoner, computing_engine=engine, callback=events.append, **kwargs), engine, events


def test_finish_is_rejected_until_missing_goals_have_evidence(react_classes):
    reasoner = ScriptedReasoner(
        [decision(), decision("finish"), decision(alpha=0.9), decision("finish")],
        [{"complete": False, "missing_goals": ["Compute the amount"], "repair_hint": "Use the successful paths"},
         {"complete": True, "missing_goals": []}],
    )
    runner, engine, events = make_runner(react_classes, reasoner)
    result = asyncio.run(runner.run("Find paths and compute the amount"))
    assert engine.run_algorithm.await_count == 2
    assert result["analysis_status"] == "completed"
    assert result["missing_goals"] == []
    assert any(event.get("content", {}).get("error") == "completion_rejected" for event in events
               if event.get("contentType") == "react_observation")
    assert "Compute the amount" in reasoner.states[2]["missing_goals"]


def test_budget_exhaustion_is_saved_and_never_emits_complete(react_classes, tmp_path):
    reasoner = ScriptedReasoner([decision()])
    runner, _, events = make_runner(react_classes, reasoner,
                                   trace_store=ReactTraceStore(tmp_path))
    runner._timed_out = MagicMock(side_effect=[False, True])
    result = asyncio.run(runner.run("Find paths and compute the amount", dataset="Example"))
    assert result["analysis_status"] == "incomplete"
    assert result["termination_reason"] == "timeout"
    assert "分析未完成" in result["analysis_result"]
    statuses = [event["content"]["stage"] for event in events if event.get("contentType") == "react_status"]
    assert statuses[-1] == "incomplete" and "complete" not in statuses
    saved = ReactTraceStore(tmp_path).get(result["run_id"])
    assert saved["status"] == "incomplete" and saved["termination_reason"] == "timeout"
    assert saved["goal"] == "Find paths and compute the amount"
    assert saved["artifacts"]["a1"]["value"] == {"score": 0.7}
    assert (tmp_path / f"{result['run_id']}.json").stat().st_mode & 0o777 == 0o600


def test_nested_numeric_error_does_not_become_evidence(react_classes):
    reasoner = ScriptedReasoner([decision(), decision("finish")])
    runner, engine, events = make_runner(react_classes, reasoner, config={"max_steps": 2})
    engine.run_algorithm.return_value = {"success": True, "result": {"error": "Invalid amount projection"}}
    result = asyncio.run(runner.run("Compute the amount"))
    assert result["analysis_status"] == "failed"
    assert not any(event.get("contentType") == "react_artifact" for event in events)
    observations = [event["content"] for event in events if event.get("contentType") == "react_observation"]
    assert observations[0]["success"] is False
    assert observations[0]["error"] == "Invalid amount projection"


def test_example_lineage_survives_generated_grouping_but_complete_field_selection_is_exact(react_classes):
    from aag.engine.react_executor import RunState
    runner, _, _ = make_runner(react_classes, None)
    state = RunState(goal="Analyze flow")
    source = runner._add_artifact(state, {"name": "temporal_paths"}, {"result": {
        "path_count": 50, "sample_paths_are_examples_only": True,
        "sample_paths": [{"nodes": ["origin", "hub", "sink"]}],
        "account_rankings": [{"node_key": "hub", "transaction_amount": 90}]}}, {}, 1)
    derived_action = {"name": "numeric_analysis", "consumed": [{"artifact_id": source}]}
    derived = runner._add_artifact(state, derived_action, {"result": {"complete_group_count": 1}}, {}, 2)
    scope = state.artifacts[derived]["evidence_scope"]
    assert runner._bounded_result({"success": True, "result": {"group_count": 1}, "evidence_scope": scope})["evidence_scope"] == scope
    assert scope["example_sources"] == [{"artifact_id": source, "field": "sample_paths", "example_count": 1, "full_path_count": 50}]
    assert runner._derived_evidence_scope({"consumed": [{"artifact_id": derived}]}, state) == scope
    assert not runner._derived_evidence_scope({"consumed": [{"artifact_id": source, "field": "account_rankings"}]}, state)
    assert state.prompt_view()["artifacts"][derived]["evidence_scope"] == scope


def test_report_is_corrected_after_scope_review_before_success(react_classes):
    reasoner = ScriptedReasoner([decision(), decision("finish")], [{"complete": True, "missing_goals": []}])
    reasoner.react_write_report = MagicMock(side_effect=["All paths have one pattern [a1].", "One representative pattern [a1]."])
    reasoner.react_review_report = MagicMock(side_effect=[
        {"valid": False, "issues": ["A sample grouping does not establish a full pattern count"]},
        {"valid": True, "issues": []}])
    runner, _, events = make_runner(react_classes, reasoner)
    result = asyncio.run(runner.run("Analyze flow with representative examples"))
    assert result["analysis_status"] == "completed"
    assert result["analysis_result"] == "One representative pattern [a1]."
    correction = reasoner.react_write_report.call_args.kwargs["execution_outcome"]["report_correction"]
    assert correction["previous_report"] == "All paths have one pattern [a1]."
    assert any(event["kind"] == "report_review" and event["data"]["valid"] is False for event in runner.events)


def test_unresolved_report_errors_cannot_emit_complete(react_classes):
    reasoner = ScriptedReasoner([decision(), decision("finish")], [{"complete": True, "missing_goals": []}])
    reasoner.react_review_report = MagicMock(return_value={"valid": False, "issues": ["Unsupported population count"]})
    runner, _, events = make_runner(react_classes, reasoner)
    result = asyncio.run(runner.run("Analyze flow"))
    assert reasoner.react_review_report.call_count == 3
    assert result["analysis_status"] == "incomplete"
    assert result["termination_reason"] == "report_failed"
    stages = [event["content"]["stage"] for event in events if event.get("contentType") == "react_status"]
    assert stages[-1] == "incomplete" and "complete" not in stages


def test_report_review_can_distinguish_full_collection_sizes_from_preview_lengths(react_classes):
    reasoner = ScriptedReasoner([decision(), decision("finish")], [{"complete": True, "missing_goals": []}])
    reasoner.react_write_report = MagicMock(return_value="Computed 20 rankings; showing examples [a1].")

    def review(**arguments):
        evidence = arguments["artifacts"]["a1"]
        assert evidence["collection_sizes"]["rankings"] == 20
        assert len(evidence["sample"]["rankings"]) == 8
        assert evidence["preview_is_truncated"] is True
        return {"valid": True, "issues": []}

    reasoner.react_review_report = review
    runner, engine, _ = make_runner(react_classes, reasoner)
    engine.run_algorithm.return_value = {"success": True, "result": {"rankings": [{"score": index} for index in range(20)]}}
    result = asyncio.run(runner.run("Rank entities"))
    assert result["analysis_status"] == "completed"


def test_structured_graph_queries_skip_model_retranslation(react_classes):
    from aag.computing_engine.computing_engine import ComputingEngine
    engine = ComputingEngine.__new__(ComputingEngine)
    engine.nl_query_engine = SimpleNamespace(
        executor=SimpleNamespace(execute=MagicMock(return_value={"success": True, "results": [], "count": 0})),
        ask=MagicMock(side_effect=AssertionError("Should not translate again")),
    )
    result = engine.execute_graph_query(query_type="neighbor_query", parameters={"label": "Account"})
    assert result["success"] is True
    engine.nl_query_engine.ask.assert_not_called()


def test_result_metadata_counts_rows_not_wrapper_fields(react_classes):
    result = react_classes._bounded_result({"success": True, "result": {"results": [], "count": 0}})
    assert result["result_total_items"] == 0
    assert result["result_field_count"] == 2


def test_named_numeric_collections_do_not_get_a_fake_record_total(react_classes):
    result = react_classes._bounded_result({"success": True, "result": {
        "primary_matches": [], "fallback_matches": [], "primary_count": 0, "fallback_count": 0,
    }})
    assert "result_total_items" not in result
    assert result["collection_sizes"] == {"primary_matches": 0, "fallback_matches": 0}


def test_numeric_process_accepts_keyword_arguments_and_defaults(react_classes):
    from aag.computing_engine.code_executor import DynamicCodeExecutor
    engine = DynamicCodeExecutor(auto_install=False)
    result = engine.execute("def process(optional=7, **inputs):\\n    return {'value': inputs['amount'] + optional}".replace("\\n", "\n"),
                            {"amount": 3}, is_numeric_analysis=True)
    assert result == {"value": 10}


def test_validator_uses_the_reasoner_adapter_method(react_classes):
    from aag.reasoner.model_deployment import Reasoner
    reasoner = Reasoner.__new__(Reasoner)
    reasoner.adapter_for = lambda _: None
    reasoner._execute_prompt = MagicMock(return_value={"is_valid": True})
    result = reasoner.nl_query_validate_cypher("RETURN 1", "Test", "node_lookup", {}, "", "", "")
    assert result["is_valid"] is True
    reasoner._execute_prompt.assert_called_once()
