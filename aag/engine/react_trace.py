"""Local, persistent execution records for inspecting failed ReAct runs."""

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from aag.utils.path_utils import RUNTIME_ROOT


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    return str(value)


_PREVIEW_KEYS = ("node_key", "vid", "id", "source_identifier", "source", "target", "src", "dst",
                 "amount", "timestamp", "properties", "results", "count")


def preview(value: Any, items: int = 8, depth: int = 5) -> Any:
    if depth <= 0:
        return str(value)[:300] if isinstance(value, (dict, list, tuple)) else json_value(value)
    if isinstance(value, Mapping):
        ordered = [key for key in _PREVIEW_KEYS if key in value]
        ordered += [key for key in value if key not in ordered]
        field_limit = max(items, 32) if len(value) <= 32 else items
        return {str(key): preview(value[key], items, depth - 1) for key in ordered[:field_limit]}
    if isinstance(value, (list, tuple, set)):
        return [preview(item, items, depth - 1) for item in list(value)[:items]]
    if isinstance(value, str):
        return value[:6000]
    return json_value(value)


def is_sampled(value: Any, items: int = 8, depth: int = 5) -> bool:
    if isinstance(value, Mapping):
        field_limit = max(items, 32) if len(value) <= 32 else items
        return depth <= 0 or len(value) > field_limit or any(is_sampled(item, items, depth - 1) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return depth <= 0 or len(value) > items or any(is_sampled(item, items, depth - 1) for item in value)
    return isinstance(value, str) and len(value) > 6000


class ReactTraceStore:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else RUNTIME_ROOT / "output" / "react-runs"

    def _path(self, run_id: str, suffix: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run id")
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{run_id}.{suffix}"

    def append(self, run_id: str, event: dict) -> None:
        path = self._path(run_id, "events.jsonl")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False) + "\n")

    def save(self, run_id: str, record: dict) -> None:
        path = self._path(run_id, "json")
        temporary = path.with_suffix(".json.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(json_value(record), stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)

    def get(self, run_id: str) -> dict:
        return json.loads(self._path(run_id, "json").read_text(encoding="utf-8"))

    def recent(self, limit: int = 30) -> list:
        if not self.directory.exists():
            return []
        records = []
        for path in sorted(self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records.append({key: data.get(key) for key in
                                ("run_id", "goal", "dataset", "status", "termination_reason", "started_at", "finished_at")})
            except (OSError, ValueError):
                continue
        return records
