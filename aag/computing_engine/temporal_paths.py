"""Bounded temporal paths with transaction identity and exact amount accounting.

The model supplies scope; this operator owns topology, chronology, deduplication
and rankings. It never infers that a transaction is illegal.
"""

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import heapq


def analyze_temporal_paths(graph, center: str, via=None, incoming_hops: int = 1,
                           outgoing_hops: int = 2, strict_time: bool = True,
                           sample_size: int = 10, max_paths: int = 100000,
                           amount_field: str = None, timestamp_field: str = None,
                           timestamp_unit: str = "iso8601", node_id_field: str = None,
                           require_path_flag: bool = False, edge_flag_field=None,
                           node_flag_field=None, flagged_nodes=None) -> dict:
    center = str(center)
    if not amount_field or not timestamp_field:
        raise ValueError("amount_field and timestamp_field must be explicitly bound to actual edge properties")
    epoch_units = {"unix_s": 1, "unix_ms": 1000, "unix_us": 1000000, "unix_ns": 1000000000}
    if timestamp_unit not in {"iso8601", *epoch_units}:
        raise ValueError("timestamp_unit must be iso8601, unix_s, unix_ms, unix_us or unix_ns")
    via = [via] if isinstance(via, str) else (via or [])
    via = [str(node) for node in via]
    if via and via[0] == center:
        via = via[1:]
    if not (0 <= incoming_hops <= 2 and 1 <= outgoing_hops <= 3 and incoming_hops + outgoing_hops <= 4):
        raise ValueError("Require 0<=incoming_hops<=2, 1<=outgoing_hops<=3 and total hops<=4")
    if not (1 <= sample_size <= 100 and 1 <= max_paths <= 500000):
        raise ValueError("Require 1<=sample_size<=100 and 1<=max_paths<=500000")
    if len(via) > outgoing_hops:
        raise ValueError("via must name successive nodes after center and fit outgoing_hops")
    vertices = {str(vertex.vid): vertex.properties for vertex in graph.vertices}
    flagged_nodes = {str(node) for node in (flagged_nodes or [])}
    node_fields = {name for props in vertices.values() for name in props}
    edge_fields = {name for edge in graph.edges for name in edge.properties}
    for name in (amount_field, timestamp_field, edge_flag_field):
        if name and name not in edge_fields:
            raise ValueError(f"Unknown edge property '{name}'; available={sorted(edge_fields)}")
    if node_flag_field and node_flag_field not in node_fields:
        raise ValueError(f"Unknown node flag property '{node_flag_field}'; available={sorted(node_fields)}")
    if node_id_field and node_id_field not in node_fields:
        raise ValueError(f"Unknown node identifier property '{node_id_field}'; available={sorted(node_fields)}")

    def canonical_identifier(value):
        value = str(value)
        if value in vertices:
            return value
        if node_id_field:
            matches = [node for node, properties in vertices.items()
                       if properties.get(node_id_field) is not None and str(properties[node_id_field]) == value]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous source identifier '{value}' in '{node_id_field}'; supply a canonical vertex ID")
        raise ValueError(f"Entity '{value}' is not a canonical vertex ID or a unique configured source identifier")

    center = canonical_identifier(center)
    via = [canonical_identifier(node) for node in via]
    if via and via[0] == center:
        via = via[1:]
    flagged_nodes = {canonical_identifier(node) for node in flagged_nodes}
    if require_path_flag and not (edge_flag_field or node_flag_field or flagged_nodes):
        raise ValueError("require_path_flag needs edge_flag_field, node_flag_field or explicit flagged_nodes")

    def positive(value):
        if isinstance(value, (float, int)):
            return value > 0
        return str(value).lower() in {"true", "yes", "1"}

    def node_flag(node):
        return node in flagged_nodes or bool(node_flag_field and positive(vertices.get(node, {}).get(node_flag_field)))
    incoming, outgoing = defaultdict(list), defaultdict(list)
    transactions = {}
    excluded = 0
    for index, edge in enumerate(graph.edges):
        try:
            amount = Decimal(str(edge.properties[amount_field]))
            timestamp = str(edge.properties[timestamp_field])
            if timestamp_unit == "iso8601":
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                moment = Decimal(str(parsed.timestamp()))
            else:
                moment = Decimal(timestamp) / epoch_units[timestamp_unit]
            if not amount.is_finite() or not moment.is_finite():
                raise ValueError("Non-finite amount or timestamp")
        except (KeyError, ValueError, TypeError, InvalidOperation, OverflowError, OSError):
            excluded += 1
            continue
        tx = {"id": index, "source": str(edge.src), "target": str(edge.dst), "amount": amount,
              "timestamp": timestamp, "moment": moment, "properties": edge.properties}
        transactions[index] = tx
        outgoing[tx["source"]].append(index)
        incoming[tx["target"]].append(index)
    if graph.edges and not transactions:
        raise ValueError("No transactions have valid amounts and timestamps; cannot infer absence of paths")

    def earlier(left, right):
        return left < right if strict_time else left <= right

    def prefixes(node, remaining, trail):
        if remaining == 0:
            yield list(reversed(trail))
            return
        for index in incoming[node]:
            if index in trail:
                continue
            if trail and not earlier(transactions[index]["moment"], transactions[trail[-1]]["moment"]):
                continue
            yield from prefixes(transactions[index]["source"], remaining - 1, trail + [index])

    def suffixes(node, remaining, trail):
        if remaining == 0:
            yield trail
            return
        for index in outgoing[node]:
            tx = transactions[index]
            if index in trail or (len(trail) < len(via) and tx["target"] != via[len(trail)]):
                continue
            if trail and not earlier(transactions[trail[-1]]["moment"], tx["moment"]):
                continue
            yield from suffixes(tx["target"], remaining - 1, trail + [index])

    before = list(prefixes(center, incoming_hops, []))
    after = list(suffixes(center, outgoing_hops, []))
    path_count, truncated = 0, False
    used = set()
    by_leg = [set() for _ in range(incoming_hops + outgoing_hops)]
    samples = []
    for left in before:
        for right in after:
            if left and not earlier(transactions[left[-1]]["moment"], transactions[right[0]]["moment"]):
                continue
            path = left + right
            if len(set(path)) != len(path):
                continue
            nodes = [transactions[path[0]]["source"], *[transactions[index]["target"] for index in path]]
            # Simple paths and simple closed cycles are allowed; internal
            # repetitions are excluded to avoid concatenating multiple loops.
            interior = nodes[:-1] if nodes[-1] == nodes[0] else nodes
            if len(set(interior)) != len(interior):
                continue
            if require_path_flag and not (any(node_flag(node) for node in nodes)
                or (edge_flag_field and any(positive(transactions[index]["properties"].get(edge_flag_field)) for index in path))):
                continue
            if path_count >= max_paths:
                truncated = True
                break
            path_count += 1
            used.update(path)
            for position, index in enumerate(path):
                by_leg[position].add(index)
            score = (transactions[right[0]]["amount"], transactions[path[-1]]["amount"])
            sample = {"nodes": nodes, "transactions": [{
                "id": index, "source": transactions[index]["source"], "target": transactions[index]["target"],
                "amount": float(transactions[index]["amount"]), "timestamp": transactions[index]["timestamp"],
                "properties": transactions[index]["properties"],
            } for index in path]}
            item = (score, path_count, sample)
            if len(samples) < sample_size:
                heapq.heappush(samples, item)
            elif item[:2] > samples[0][:2]:
                heapq.heapreplace(samples, item)
        if truncated:
            break

    def total(indices):
        return float(sum((transactions[index]["amount"] for index in indices), Decimal("0")).quantize(Decimal("0.01")))

    accounts = {}
    for index in used:
        tx = transactions[index]
        for node, direction in [(tx["source"], "outgoing"), (tx["target"], "incoming")]:
            if node not in accounts:
                accounts[node] = {"node_key": node,
                                  "incoming": set(), "outgoing": set(), "max_single_amount": Decimal("0")}
            accounts[node][direction].add(index)
            accounts[node]["max_single_amount"] = max(accounts[node]["max_single_amount"], tx["amount"])
    ranking = []
    for info in accounts.values():
        ranking.append({"node_key": info["node_key"],
                        "source_identifier": {"field": node_id_field,
                                              "value": vertices.get(info["node_key"], {}).get(node_id_field)} if node_id_field else None,
                        "has_node_flag": node_flag(info["node_key"]) if node_flag_field or flagged_nodes else None,
                        "incoming_amount": total(info["incoming"]), "outgoing_amount": total(info["outgoing"]),
                        "transaction_amount": total(info["incoming"] | info["outgoing"]),
                        "max_single_amount": float(info["max_single_amount"]),
                        "unique_transaction_count": len(info["incoming"] | info["outgoing"])})
    ranking.sort(key=lambda row: (-row["transaction_amount"], row["node_key"]))
    max_single = max((row["max_single_amount"] for row in ranking), default=0)
    return {
        "center": center, "via": via, "scope": {"incoming_hops": incoming_hops, "outgoing_hops": outgoing_hops,
            "strict_time": strict_time, "amount_field": amount_field, "timestamp_field": timestamp_field,
            "timestamp_unit": timestamp_unit, "node_id_field": node_id_field,
            "hop_semantics": "Exactly incoming_hops before center and exactly outgoing_hops after center; shorter lengths are separate scopes",
            "path_topology": "Simple paths or simple closed cycles; no repeated internal vertices",
            "naive_timestamp_timezone": "UTC", "require_path_flag": require_path_flag,
            "edge_flag_field": edge_flag_field, "node_flag_field": node_flag_field,
            "explicit_flagged_nodes": sorted(flagged_nodes),
            "node_flag_evaluation": "node_flag_field is evaluated on node properties independently of the explicit flagged-node list; an empty explicit list does not mean every node flag is false",
            "flag_rule": "At least one path edge or node has a supplied flag; dataset flags are screening proxies, not proof of a business conclusion"},
        "path_count": path_count, "path_count_is_exact": not truncated, "truncated": truncated,
        "sample_paths_are_examples_only": True,
        "refinement_instruction": "Re-run temporal_paths with screening flags/via parameters to change the analyzed set. Do not re-filter sample_paths and treat the examples as the complete path set.",
        "sample_paths": [item[2] for item in sorted(samples, key=lambda item: item[:2], reverse=True)],
        "unique_transaction_count": len(used),
        "qualified_center_outflow_amount": total(by_leg[incoming_hops]),
        "qualified_center_outflow_count": len(by_leg[incoming_hops]),
        "all_center_outflow_amount": total(outgoing[center]),
        "unique_amount_by_leg": [{"position": index + 1, "amount": total(indices), "count": len(indices)}
                                 for index, indices in enumerate(by_leg)],
        "all_path_transaction_volume": total(used), "account_rankings": ranking,
        "largest_total_amount_account": ranking[0] if ranking else None,
        "max_single_amount": max_single,
        "largest_single_transaction_accounts": [row for row in ranking if row["max_single_amount"] == max_single],
        "excluded_invalid_transactions": excluded,
        "amount_basis": "Unique original transaction indices, including parallel transfers. Center outflows are counted once at the center-outgoing leg. Account rankings aggregate each actual transaction once per account; they include the center. Total path volume is separate from origin outflows. These are scoped flow amounts, not a finding of illegality.",
    }
