from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class ToolDocument:
    id: str
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    server: str | None = None
    category: str | None = None

    def retrieval_text(self) -> str:
        """Return a deterministic text projection used by retrievers."""
        parts = [self.name, self.description]
        if self.server:
            parts.append(self.server)
        if self.category:
            parts.append(self.category)
        if self.parameters:
            parts.append(json.dumps(self.parameters, ensure_ascii=False, sort_keys=True))
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class QueryExample:
    id: str
    query: str
    labels: frozenset[str]
    instruction: str | None = None

    @property
    def search_text(self) -> str:
        if self.instruction:
            return f"{self.instruction}\n{self.query}"
        return self.query


@dataclass(frozen=True)
class RankedTool:
    tool_id: str
    score: float
    rank: int
