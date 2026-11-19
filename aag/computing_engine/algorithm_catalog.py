"""Compact model-facing contracts built from connected, callable algorithm tools."""

import re


def _text(value, limit):
    return " ".join(str(value or "").split())[:limit]


def algorithm_tool_catalog(runtime_tools, knowledge_index, limit=120):
    """Runtime registration/schema are authoritative; knowledge adds task context.

    Do not publish knowledge-only algorithms, stale parameter names or static
    graph-type claims. The connected implementation supplies its own behavior.
    """
    catalog = []
    for runtime in runtime_tools:
        algorithm_id = runtime["id"]
        info = knowledge_index.get(algorithm_id, {}) or {}
        schema = runtime.get("input_schema") or {}
        parameters = schema.get("parameters") or schema.get("properties") or {}
        supplied = set(runtime.get("system_supplied_parameters", []))
        required = set(schema.get("required") or []) - supplied
        parameter_schema = {}
        for name, value in parameters.items():
            if name in supplied:
                continue
            value = value if isinstance(value, dict) else {"type": str(value)}
            if value.get("required") is True:
                required.add(name)
            parameter_schema[name] = {key: value[key] for key in ("type", "default", "enum") if key in value}
            parameter_schema[name]["description"] = _text(value.get("description"), 140 if name in required else 80)
        # The server description comes from the implementation's docstring.
        description = re.split(r"\n\s*(?:Parameters|Args|Returns|Examples)\s*\n", runtime.get("description") or "", maxsplit=1)[0]
        output = runtime.get("output_schema") or {}
        catalog.append({
            "id": algorithm_id, "type": "algorithm", "name": algorithm_id,
            "engine": runtime.get("engine"), "tool_name": runtime.get("tool_name"),
            "description": _text(description, 500),
            "category": info.get("task_type_id"),
            "required_parameters": sorted(required),
            "optional_parameters": [name for name in parameter_schema if name not in required],
            "parameter_schema": parameter_schema,
            "system_supplied_parameters": sorted(supplied),
            "output": {"type": output.get("type"), "description": _text(output.get("description"), 220)},
            "implementation_behavior": {key: value for key, value in runtime.get("implementation_behavior", {}).items()
                                        if key not in {"graph", "result_serialization"}},
        })
    return catalog[:limit]
