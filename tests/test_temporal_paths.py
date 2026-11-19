from types import SimpleNamespace

import pytest

from aag.computing_engine.temporal_paths import analyze_temporal_paths

FIELDS = {"amount_field": "base_amt", "timestamp_field": "tran_timestamp", "node_id_field": "acct_id"}


def analyze(data, **parameters):
    return analyze_temporal_paths(data, **(FIELDS | parameters))


def graph():
    names = ["Upstream", "Center", "Middle", "Wrong", "Downstream"]
    vertices = [SimpleNamespace(vid=name, properties={"acct_id": index}) for index, name in enumerate(names)]
    raw = [("Upstream", "Center", 100, 1), ("Center", "Middle", 90, 2),
           ("Center", "Wrong", 999, 2), ("Middle", "Downstream", 80, 3),
           ("Center", "Middle", 90, 2)]
    edges = [SimpleNamespace(src=source, dst=target, properties={
        "base_amt": amount, "tran_timestamp": f"2025-01-0{day}T00:00:00Z"})
        for source, target, amount, day in raw]
    return SimpleNamespace(vertices=vertices, edges=edges)


def test_temporal_join_requires_connected_accounts_and_deduplicates_shared_legs():
    result = analyze(graph(), center="Center", via=["Middle"])
    assert result["path_count"] == 2
    assert result["qualified_center_outflow_amount"] == 180
    assert result["qualified_center_outflow_count"] == 2  # identical parallel transfers remain distinct
    assert result["all_center_outflow_amount"] == 1179
    assert result["unique_amount_by_leg"][-1]["amount"] == 80  # shared downstream transfer counted once
    assert result["largest_total_amount_account"]["node_key"] == "Center"
    middle = next(row for row in result["account_rankings"] if row["node_key"] == "Middle")
    assert middle["incoming_amount"] == 180  # not multiplied by number of path combinations
    assert all("Wrong" not in path["nodes"] for path in result["sample_paths"])
    for path in result["sample_paths"]:
        for left, right in zip(path["transactions"], path["transactions"][1:]):
            assert left["target"] == right["source"]
            assert left["timestamp"] < right["timestamp"]


def test_nonchronological_edges_are_not_paths():
    data = graph()
    data.edges[3].properties["tran_timestamp"] = "2024-12-31T00:00:00Z"
    assert analyze(data, center="Center", via=["Middle"])["path_count"] == 0


def test_configured_source_identifiers_resolve_to_canonical_ids_only_when_unique():
    result = analyze(graph(), center=1, via=[2], require_path_flag=True, flagged_nodes=[2])
    assert result["center"] == "Center" and result["via"] == ["Middle"]
    assert result["path_count"] == 2 and result["scope"]["explicit_flagged_nodes"] == ["Middle"]
    data = graph()
    data.vertices[-1].properties["acct_id"] = 1
    with pytest.raises(ValueError, match="Ambiguous source identifier"):
        analyze(data, center=1)


def test_capped_path_count_is_explicitly_inexact():
    result = analyze(graph(), center="Center", via=["Middle"], max_paths=1)
    assert result["truncated"] is True
    assert result["path_count_is_exact"] is False
    assert result["qualified_center_outflow_amount"] == 90


def test_wrong_amount_field_cannot_look_like_zero_outflow():
    with pytest.raises(ValueError, match="base_amt"):
        analyze(graph(), center="Center", amount_field="rel_base_amt")


def test_screening_filter_runs_on_all_paths_not_only_the_sample():
    data = graph()
    for vertex in data.vertices:
        vertex.properties["prior_sar_count"] = vertex.vid == "Middle"
    result = analyze(data, center="Center", via=["Middle"], sample_size=1,
                                    require_path_flag=True, node_flag_field="prior_sar_count")
    assert len(result["sample_paths"]) == 1
    assert result["path_count"] == 2
    assert result["qualified_center_outflow_amount"] == 180  # full set, not just the 90 in the sample
    assert result["scope"]["require_path_flag"] is True


def test_no_screening_match_is_zero_only_after_valid_field_check():
    data = graph()
    for edge in data.edges:
        edge.properties["is_sar"] = False
    result = analyze(data, center="Center", require_path_flag=True, edge_flag_field="is_sar")
    assert result["path_count"] == 0
    with pytest.raises(ValueError, match="Unknown node flag"):
        analyze(data, center="Center", require_path_flag=True, node_flag_field="is_mule")


def test_dataset_field_names_are_never_implicit_defaults():
    with pytest.raises(ValueError, match="explicitly bound"):
        analyze_temporal_paths(graph(), center="Center")


@pytest.mark.parametrize("unit,scale", [("iso8601", None), ("unix_s", 1), ("unix_ms", 1000),
                                       ("unix_us", 1000000), ("unix_ns", 1000000000)])
def test_renamed_fields_and_timestamp_encodings_preserve_flow_results(unit, scale):
    from datetime import datetime
    original, renamed = graph(), graph()
    for vertex in renamed.vertices:
        vertex.properties = {"external_ref": vertex.properties["acct_id"], "reviewed": vertex.vid == "Middle"}
    for edge in renamed.edges:
        stamp = edge.properties["tran_timestamp"]
        edge.properties = {"quantity": edge.properties["base_amt"],
                           "happened_at": int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()) * scale if scale else stamp,
                           "marked": False}
    expected = analyze(original, center="Center", via=["Middle"])
    result = analyze_temporal_paths(renamed, center="Center", amount_field="quantity", timestamp_field="happened_at",
                                    timestamp_unit=unit, node_id_field="external_ref", require_path_flag=True,
                                    node_flag_field="reviewed", edge_flag_field="marked")
    for key in ("path_count", "qualified_center_outflow_amount", "qualified_center_outflow_count",
                "all_center_outflow_amount", "all_path_transaction_volume", "unique_transaction_count", "unique_amount_by_leg"):
        assert result[key] == expected[key]
    assert [(row["node_key"], row["transaction_amount"]) for row in result["account_rankings"]] == [
        (row["node_key"], row["transaction_amount"]) for row in expected["account_rankings"]]
    assert result["largest_total_amount_account"]["source_identifier"] == {"field": "external_ref", "value": 1}


def test_invalid_timestamp_encoding_is_an_error_not_an_absence_of_paths():
    data = graph()
    for edge in data.edges:
        edge.properties["tran_timestamp"] = "not-a-time"
    with pytest.raises(ValueError, match="No transactions have valid"):
        analyze(data, center="Center")
    with pytest.raises(ValueError, match="timestamp_unit must be"):
        analyze(graph(), center="Center", timestamp_unit="auto")
