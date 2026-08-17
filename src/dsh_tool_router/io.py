from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import QueryExample, ToolDocument


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_tools(path: str | Path) -> list[ToolDocument]:
    tools: list[ToolDocument] = []
    for row in _read_jsonl(Path(path)):
        tools.append(
            ToolDocument(
                id=str(row["id"]),
                name=str(row.get("name", row["id"])),
                description=str(row.get("description", "")),
                parameters=dict(row.get("parameters", {})),
                server=_optional_string(row.get("server")),
                category=_optional_string(row.get("category")),
            )
        )
    return tools


def load_queries(path: str | Path) -> list[QueryExample]:
    examples: list[QueryExample] = []
    for row in _read_jsonl(Path(path)):
        labels = row.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"query {row.get('id')!r} must contain non-empty labels")
        examples.append(
            QueryExample(
                id=str(row["id"]),
                query=str(row["query"]),
                labels=frozenset(str(label) for label in labels),
                instruction=_optional_string(row.get("instruction")),
            )
        )
    return examples


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
