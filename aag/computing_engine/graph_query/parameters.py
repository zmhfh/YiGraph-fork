"""Normalize graph tool arguments against the database's actual schema."""

from copy import deepcopy
import re
from typing import Any, Dict, Optional


def normalize_ordering(params: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(params)
    order = result.get("order_by")
    direction = result.get("order_direction", "ASC")
    if isinstance(order, dict):
        if "field" in order:
            direction = order.get("direction", direction)
            order = order["field"]
        elif len(order) == 1:
            order, direction = next(iter(order.items()))
        elif not order:
            order = None
        else:
            raise ValueError("order_by must contain field/direction or one field:direction pair")
    if isinstance(order, str):
        parts = order.split()
        if len(parts) == 2 and parts[1].upper() in {"ASC", "DESC"}:
            order, direction = parts[0], parts[1]
    direction = str(direction).upper()
    if direction not in {"ASC", "DESC"}:
        raise ValueError("order_direction must be ASC or DESC")
    if "order_by" in result:
        result["order_by"] = order
    if order or "order_direction" in result:
        result["order_direction"] = direction
    return result


def normalize_query_parameters(query_type: str, params: Dict[str, Any], schema: Optional[dict]) -> Dict[str, Any]:
    """Resolve casing/aliases and reject nonexistent properties before querying.

    A missing property used to silently return null, which looked like an
    absence of transaction data. Here it is a repairable tool input error.
    """
    result = normalize_ordering(params)
    if "direction" in result:
        direction = str(result["direction"]).lower()
        result["direction"] = {"outgoing": "out", "outbound": "out", "incoming": "in", "inbound": "in"}.get(direction, direction)
        if result["direction"] not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in or both")
    allowed = {
        "node_lookup": {"label", "key", "value", "conditions", "return_fields", "order_by", "order_direction", "limit", "aggregate", "aggregate_field"},
        "neighbor_query": {"label", "key", "value", "hops", "rel_type", "direction", "return_fields", "order_by", "order_direction", "limit", "where", "return_distinct", "exclude_start", "return_path_length", "aggregate", "aggregate_field"},
        "relationship_filter": {"rel_type", "start_label", "end_label", "rel_conditions", "return_fields", "aggregate", "aggregate_field", "order_by", "order_direction", "limit"},
        "subgraph": {"label", "key", "value", "hops", "rel_type", "direction", "where", "limit_paths", "rel_conditions", "start_label", "end_label", "limit", "aggregate", "aggregate_field", "return_fields"},
        "subgraph_by_nodes": {"label", "key", "values", "include_internal", "rel_type", "direction", "where", "aggregate", "aggregate_field", "return_fields"},
    }.get(query_type)
    if allowed and set(result) - allowed:
        raise ValueError(f"Unsupported {query_type} parameters: {sorted(set(result) - allowed)}; allowed={sorted(allowed)}. Use neighbor_query with label/key/value to filter transactions around a named entity.")
    if query_type == "subgraph" and "limit" in result and not result.get("rel_conditions"):
        raise ValueError("Center-based subgraph uses limit_paths, not limit; returned path-limit metadata describes whether the candidate scope is complete")
    if query_type == "path_query":
        result.setdefault("max_hops", 4)
        result.setdefault("limit", 50)
    if query_type == "neighbor_query" and isinstance(result.get("where"), str):
        result["where"] = re.sub(r"\b(?:rel|firstRel)\.", "r.", result["where"])
        result["where"] = re.sub(r"\bneighbor\.", "nbr.", result["where"])
    if isinstance(result.get("rel_conditions"), list):
        conditions = {}
        for item in result["rel_conditions"]:
            if not isinstance(item, dict) or not item.get("field") or "value" not in item:
                raise ValueError("rel_conditions must be an object property->value or property->[operator,value]")
            name = item["field"]
            if name in conditions:
                raise ValueError(f"Repeated condition for '{name}'; use explicit range bounds")
            conditions[name] = [item.get("operator", item.get("op", "=")), item["value"]]
        result["rel_conditions"] = conditions
    if not schema:
        return result
    labels = schema.get("node_labels", {})
    relationships = schema.get("relationship_types", {})

    def canonical(value, choices, kind):
        if value in choices:
            return value
        matches = [name for name in choices if str(name).lower() == str(value).lower()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown {kind} '{value}'; available: {list(choices)}")

    for key in ("label", "start_label", "end_label", "node_label"):
        if result.get(key):
            result[key] = canonical(result[key], labels, "node label")
    if result.get("rel_type"):
        result["rel_type"] = canonical(result["rel_type"], relationships, "relationship type")
    label = result.get("label") or result.get("node_label")
    node_fields = set(labels.get(label, {}).get("properties", [])) if label else {
        name for info in labels.values() for name in info.get("properties", [])
    }
    rel_fields = set(relationships.get(result.get("rel_type"), {}).get("properties", [])) if result.get("rel_type") else {
        name for info in relationships.values() for name in info.get("properties", [])
    }
    if isinstance(result.get("rel_conditions"), dict):
        normalized = {}
        for name, value in result["rel_conditions"].items():
            name = re.sub(r"^(?:rel|r|t)\.", "", name)
            field_name = name.rsplit("_", 1)[0] if name.endswith(("_start", "_end")) else name
            if field_name not in rel_fields:
                raise ValueError(f"Unknown relationship property '{field_name}'; available={sorted(rel_fields)}")
            normalized[name] = value
        result["rel_conditions"] = normalized
    if result.get("key") == "name" and "name" not in node_fields and "node_key" in node_fields:
        result["key"] = "node_key"
    if result.get("key") and result["key"] not in node_fields:
        raise ValueError(f"Unknown node key '{result['key']}'; available: {sorted(node_fields)}")
    if query_type == "path_query":
        computed = {str(name).lower() for name in result.get("return_fields") or []} & {
            "totalamount", "maxamount", "minamount", "avgamount"}
        if computed and not result.get("amount_field"):
            raise ValueError("Computed path amounts require amount_field set to an actual relationship property")
        if result.get("amount_field") and result["amount_field"] not in rel_fields:
            raise ValueError(f"Unknown relationship amount property '{result['amount_field']}'; available={sorted(rel_fields)}")

    def field(value, neighbor=False):
        value = str(value)
        if "." not in value and value not in node_fields and value not in rel_fields:
            for prefix in ("neighbor", "start", "nbr", "rel", "from", "to", "r", "n"):
                if value.startswith(prefix + "_") and value[len(prefix) + 1:] in node_fields | rel_fields:
                    value = prefix + "." + value[len(prefix) + 1:]
                    break
        if any(token in value for token in ("(", ")", " AS ", ";")):
            raise ValueError("return_fields accepts property references, not expressions or AS aliases; use numeric_analysis for calculations")
        if "." in value:
            prefix, prop = value.split(".", 1)
            prefix = {"a": "start", "b": "nbr" if neighbor else "end", "from": "start", "to": "end", "t": "rel", "r": "rel",
                      "firstRel": "rel", "neighbor": "nbr"}.get(prefix, prefix)
        else:
            prop = value
            prefix = "rel" if prop in rel_fields and prop not in node_fields else ("nbr" if neighbor else "n")
        fields = rel_fields if prefix == "rel" else node_fields
        if prop not in fields:
            raise ValueError(f"Unknown {prefix} property '{prop}'; node properties={sorted(node_fields)}, relationship properties={sorted(rel_fields)}")
        if query_type == "node_lookup":
            return prop
        if query_type == "relationship_filter":
            if prefix == "n":
                raise ValueError("Use from./to. prefixes for endpoint properties and rel. for relationship properties")
            prefix = {"start": "from", "end": "to"}.get(prefix, prefix)
            if prefix not in {"from", "to", "rel"}:
                raise ValueError("relationship_filter fields use from./to./rel. prefixes")
            return f"{prefix}.{prop}"
        if prefix not in ({"start", "nbr", "rel"} if neighbor else {"start", "end", "rel", "n"}):
            raise ValueError(f"Unknown field prefix '{prefix}'; use start/nbr/rel for a neighbor query")
        return f"{prefix}.{prop}"

    if query_type in {"node_lookup", "neighbor_query", "relationship_filter"}:
        if result.get("return_fields"):
            result["return_fields"] = [field(value, query_type == "neighbor_query") for value in result["return_fields"]]
        if result.get("order_by"):
            order = field(result["order_by"], query_type == "neighbor_query")
            if query_type == "relationship_filter":
                if not order.startswith("rel."):
                    raise ValueError("relationship_filter can order by relationship properties; use numeric_analysis for endpoint rankings")
                order = order[4:]
            if query_type == "neighbor_query" and result.get("hops", 1) == 1:
                if result.get("return_fields") and order not in result["return_fields"]:
                    result["return_fields"].append(order)
                order = order.replace(".", "_") if result.get("return_fields") else order
            result["order_by"] = order
    return result
