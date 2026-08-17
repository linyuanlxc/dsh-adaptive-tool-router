from __future__ import annotations

import ast
import json
from typing import Any, Iterable

from .models import QueryExample, ToolDocument


TOOLRET_TASK_TO_CATEGORY = {
    "craft-math-algebra": "code",
    "craft-tabmwp": "code",
    "craft-vqa": "code",
    "gorilla-huggingface": "code",
    "gorilla-pytorch": "code",
    "gorilla-tensor": "code",
    "toolink": "code",
    "apibank": "web",
    "apigen": "web",
    "mnms": "web",
    "reversechain": "web",
    "rotbench": "web",
    "t-eval-dialog": "web",
    "t-eval-step": "web",
    "taskbench-daily": "web",
    "toolace": "web",
    "toolbench": "web",
    "toolemu": "web",
    "tooleyes": "web",
    "toollens": "web",
    "ultratool": "web",
    "autotools-food": "web",
    "autotools-music": "web",
    "autotools-weather": "web",
    "restgpt-spotify": "web",
    "restgpt-tmdb": "web",
    "appbench": "customized",
    "gpt4tools": "customized",
    "gta": "customized",
    "taskbench-huggingface": "customized",
    "taskbench-multimedia": "customized",
    "metatool": "customized",
    "tool-be-honest": "customized",
    "toolalpaca": "customized",
    "toolbench-sam": "customized",
}


def load_toolret(
    task: str,
    *,
    all_categories: bool = True,
) -> tuple[list[ToolDocument], list[QueryExample]]:
    """Load one ToolRet task through the official Hugging Face datasets."""
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as error:
        raise RuntimeError(
            'ToolRet support requires: pip install -e ".[toolret]"'
        ) from error

    if task not in TOOLRET_TASK_TO_CATEGORY:
        supported = ", ".join(sorted(TOOLRET_TASK_TO_CATEGORY))
        raise ValueError(f"unknown ToolRet task {task!r}; choose one of: {supported}")

    categories = (
        sorted(set(TOOLRET_TASK_TO_CATEGORY.values()))
        if all_categories
        else [TOOLRET_TASK_TO_CATEGORY[task]]
    )
    raw_tools = concatenate_datasets(
        [load_dataset("mangopy/ToolRet-Tools", category)["tools"] for category in categories]
    )
    raw_queries = load_dataset("mangopy/ToolRet-Queries", task)["queries"]

    tools = [_convert_tool(row) for row in raw_tools]
    queries = [_convert_query(row) for row in raw_queries]
    return tools, queries


def _convert_tool(row: dict[str, Any]) -> ToolDocument:
    documentation = _decode_object(row.get("doc", row.get("documentation", {})))
    if not isinstance(documentation, dict):
        documentation = {"description": str(documentation)}
    raw_parameters = documentation.get(
        "parameters",
        documentation.get("doc_arguments", {}),
    )
    parameters = (
        raw_parameters
        if isinstance(raw_parameters, dict)
        else {"raw": raw_parameters}
    )
    tool_id = str(row["id"])
    return ToolDocument(
        id=tool_id,
        name=str(documentation.get("name", tool_id)),
        description=str(documentation.get("description", row.get("documentation", ""))),
        parameters=parameters,
        category=_optional_string(row.get("category")),
    )


def _convert_query(row: dict[str, Any]) -> QueryExample:
    labels = _decode_object(row["labels"])
    if not isinstance(labels, Iterable) or isinstance(labels, (str, bytes, dict)):
        raise ValueError(f"query {row.get('id')!r} has invalid labels")
    label_ids = frozenset(str(label["id"]) for label in labels)
    return QueryExample(
        id=str(row["id"]),
        query=str(row["query"]),
        labels=label_ids,
        instruction=_optional_string(row.get("instruction")),
    )


def _decode_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
