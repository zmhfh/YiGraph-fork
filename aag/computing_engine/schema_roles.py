"""Bind semantic tool arguments from dataset declarations, never column-name guesses."""

from collections.abc import Mapping


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def temporal_path_schema(dataset):
    """Expose declared roles and only auto-bind a single, unambiguous schema.

    Role declarations are metadata. Actual property existence and values are
    checked by the operator when called, including explicit model overrides.
    Screening fields are choices for the model, not automatically enabled filters.
    """
    schema = _get(dataset, "schema")
    vertices = _get(schema, "vertex", []) or []
    edges = _get(schema, "edge", []) or []
    defaults, screening = {}, {}
    if len(edges) == 1:
        roles = _get(edges[0], "semantic_roles", {}) or {}
        for role, parameter in (("value", "amount_field"), ("timestamp", "timestamp_field"),
                                ("timestamp_unit", "timestamp_unit")):
            if roles.get(role):
                defaults[parameter] = roles[role]
        if roles.get("screen_flag"):
            screening["edge_flag_field"] = roles["screen_flag"]
    if len(vertices) == 1:
        identifier = _get(vertices[0], "id_field")
        if identifier:
            defaults["node_id_field"] = identifier
        roles = _get(vertices[0], "semantic_roles", {}) or {}
        if roles.get("screen_flag"):
            screening["node_flag_field"] = roles["screen_flag"]
    return {
        "defaults": defaults,
        "declared_screening_fields": screening,
        "vertex_schemas": [{"type": _get(item, "type"), "id_field": _get(item, "id_field"),
                            "semantic_roles": _get(item, "semantic_roles", {}) or {}} for item in vertices],
        "edge_schemas": [{"type": _get(item, "type"),
                          "semantic_roles": _get(item, "semantic_roles", {}) or {}} for item in edges],
        "binding_rule": "Defaults come only from a single declared schema. Missing or ambiguous value/time roles require explicit actual property names. Select screening fields explicitly for the requested scope; flags do not establish business conclusions.",
    }


def bind_temporal_path_parameters(dataset, parameters):
    bound = dict(temporal_path_schema(dataset)["defaults"])
    bound.update(parameters)
    missing = [name for name in ("amount_field", "timestamp_field") if not bound.get(name)]
    if missing:
        raise ValueError(f"Missing temporal-path field bindings: {missing}. Declare semantic_roles in the dataset schema or supply actual property names explicitly; no dataset-specific fields are assumed.")
    return bound
