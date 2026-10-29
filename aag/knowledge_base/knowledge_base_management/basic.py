from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


from aag.utils.path_utils import KNOWLEDGE_BASE_DIR

BASE_DIR = KNOWLEDGE_BASE_DIR
ALGORITHMS_PATH = BASE_DIR / "algorithms.yaml"
TASK_TYPES_PATH = BASE_DIR / "task_types.yaml"
DEFAULT_LANG = "zh-CN"


def get_algorithm_list(
    search: str | None = None,
    category: str | None = None,
    lang: str = DEFAULT_LANG,
) -> dict[str, Any]:
    algorithms = [_format_algorithm(item, lang) for item in _load_algorithms()]

    if category:
        category_key = _normalize_token(category)
        algorithms = [
            item for item in algorithms
            if _normalize_token(item["category"]) == category_key
        ]

    if search:
        search_key = search.casefold().strip()
        algorithms = [
            item for item in algorithms
            if _algorithm_matches_search(item, search_key)
        ]

    return {
        "success": True,
        "data": algorithms,
        "total": len(algorithms),
    }


def get_category_list(lang: str = DEFAULT_LANG) -> dict[str, Any]:
    categories = []
    algorithm_counts = _count_algorithms_by_category()

    for item in _load_task_types():
        category_id = str(item.get("id", "")).strip()
        if not category_id:
            continue

        categories.append({
            "id": category_id,
            "label": _localize_text(item.get("task_type") or category_id, lang),
            "count": algorithm_counts.get(category_id, 0),
        })

    return {
        "success": True,
        "data": categories,
    }


def get_algorithm_detail(algorithm_id: str, lang: str = DEFAULT_LANG) -> dict[str, Any] | None:
    normalized_id = _normalize_algorithm_id(algorithm_id)

    for item in _load_algorithms():
        if item.get("id") == normalized_id:
            return {
                "success": True,
                "data": _format_algorithm(item, lang),
            }

    return None


def format_algorithm_record(item: dict[str, Any], lang: str = DEFAULT_LANG) -> dict[str, Any]:
    return _format_algorithm(item, lang)


def clear_knowledge_base_cache() -> None:
    _load_algorithms.cache_clear()
    _load_task_types.cache_clear()


@lru_cache(maxsize=1)
def _load_algorithms() -> tuple[dict[str, Any], ...]:
    return tuple(_load_yaml_list(ALGORITHMS_PATH))


@lru_cache(maxsize=1)
def _load_task_types() -> tuple[dict[str, Any], ...]:
    return tuple(_load_yaml_list(TASK_TYPES_PATH))


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or []

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in knowledge base file: {path}")

    return [item for item in payload if isinstance(item, dict)]


def _format_algorithm(item: dict[str, Any], lang: str) -> dict[str, Any]:
    algorithm_name = str(item.get("id", "")).strip()
    deployment = _as_dict(item.get("Deployment_method"))
    principles = _as_dict(item.get("Principles"))
    input_schema = deployment.get("input_schema")
    output_schema = deployment.get("output_schema")
    category = str(item.get("task_type_id", "")).strip()
    principle_text = (
        principles.get("description")
        or item.get("Principles")
        or item.get("Application_scenario")
        or ""
    )

    return {
        "id": _api_algorithm_id(algorithm_name),
        "name": algorithm_name,
        "displayName": _localize_text(_humanize_algorithm_name(algorithm_name), lang),
        "category": category,
        "summary": _localize_text(_first_sentence(item.get("Application_scenario")), lang),
        "description": _localize_text(principle_text, lang),
        "complexity": principles.get("time_complexity") or "",
        "inputs": _format_input_schema(input_schema, lang),
        "outputs": _format_output_schema(output_schema, lang),
        "useCases": _localize_list(item.get("solvable_questions") or [], lang),
        "chartPreference": _infer_chart_preference(output_schema),
        "tags": _build_tags(item),
    }


def _format_input_schema(input_schema: Any, lang: str) -> list[str]:
    if isinstance(input_schema, str):
        text = _localize_text(input_schema, lang).strip()
        return [text] if text else []

    parameters = _as_dict(_as_dict(input_schema).get("parameters"))
    return _format_inputs(parameters, lang)


def _format_inputs(parameters: dict[str, Any], lang: str) -> list[str]:
    inputs = []

    for name, meta in parameters.items():
        if not isinstance(meta, dict):
            inputs.append(str(name))
            continue

        details = []
        param_type = meta.get("type")
        if param_type:
            details.append(str(param_type))
        if meta.get("required") is True:
            details.append("required")
        elif "default" in meta:
            details.append(f"default={meta.get('default')}")

        description = _localize_text(meta.get("description") or "", lang)
        suffix = f" ({', '.join(details)})" if details else ""
        separator = " — " if description else ""
        inputs.append(f"{name}{separator}{description}{suffix}")

    return inputs


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_outputs(output_schema: dict[str, Any], lang: str) -> list[str]:
    if isinstance(output_schema, str):
        text = _localize_text(output_schema, lang).strip()
        return [text] if text else []

    if not output_schema:
        return []

    output_type = output_schema.get("type")
    description = _localize_text(output_schema.get("description") or "", lang)
    if output_type and description:
        return [f"{description} ({output_type})"]
    if description:
        return [description]
    if output_type:
        return [str(output_type)]
    return []


def _build_tags(item: dict[str, Any]) -> list[str]:
    return _dedupe([
        str(value).strip()
        for value in (item.get("id"), item.get("task_type_id"))
        if str(value or "").strip()
    ])


def _count_algorithms_by_category() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _load_algorithms():
        category = str(item.get("task_type_id", "")).strip()
        if category:
            counts[category] = counts.get(category, 0) + 1
    return counts


def _algorithm_matches_search(item: dict[str, Any], search_key: str) -> bool:
    fields = [
        item.get("name", ""),
        item.get("displayName", ""),
        *item.get("tags", []),
    ]
    return any(search_key in str(value).casefold() for value in fields)


def _normalize_algorithm_id(algorithm_id: str) -> str:
    normalized = str(algorithm_id).strip()
    if normalized.startswith("run_"):
        normalized = normalized[4:]
    return normalized


def _api_algorithm_id(algorithm_name: str) -> str:
    return f"run_{algorithm_name}"


def _normalize_token(value: str) -> str:
    return str(value).replace("-", " ").replace("_", " ").casefold().strip()


def _humanize_algorithm_name(name: str) -> str:
    special_names = {
        "bfs": "BFS",
        "dfs": "DFS",
        "hits": "HITS",
        "pagerank": "PageRank",
    }
    if name in special_names:
        return special_names[name]

    return " ".join(part.upper() if len(part) <= 3 else part.capitalize() for part in name.split("_"))


def _first_sentence(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    for delimiter in (". ", "? ", "! "):
        if delimiter in text:
            return text.split(delimiter, 1)[0].strip() + delimiter.strip()
    return text


def _localize_text(value: Any, lang: str) -> str:
    if isinstance(value, dict):
        return str(value.get(lang) or value.get(DEFAULT_LANG) or value.get("en-US") or next(iter(value.values()), ""))
    return str(value or "")


def _localize_list(values: list[Any], lang: str) -> list[str]:
    return [_localize_text(value, lang) for value in values]


def _format_output_schema(output_schema: Any, lang: str) -> list[str]:
    return _format_outputs(output_schema, lang)


def _infer_chart_preference(output_schema: Any) -> str:
    if isinstance(output_schema, str):
        combined = output_schema.casefold()
    else:
        output_schema = _as_dict(output_schema)
        output_type = str(output_schema.get("type", "")).casefold()
        description = str(output_schema.get("description", "")).casefold()
        combined = f"{output_type} {description}"

    if "dictionary" in combined or "dict" in combined:
        return "bar"
    if "list" in combined or "set" in combined or "tuple" in combined:
        return "table"
    if "path" in combined:
        return "line"
    return "table"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
