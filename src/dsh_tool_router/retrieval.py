from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable, Protocol

from .models import RankedTool, ToolDocument


_TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


class ToolRetriever(Protocol):
    def rank(self, query: str, *, limit: int = 10) -> list[RankedTool]:
        """Return tools ordered by decreasing relevance."""


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


class BM25Router:
    """Small dependency-free BM25 baseline for reproducible evaluation."""

    def __init__(
        self,
        tools: Iterable[ToolDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tools = tuple(tools)
        if not self.tools:
            raise ValueError("at least one tool is required")
        if len({tool.id for tool in self.tools}) != len(self.tools):
            raise ValueError("tool ids must be unique")

        self.k1 = k1
        self.b = b
        self._documents = [tokenize(tool.retrieval_text()) for tool in self.tools]
        self._term_counts = [Counter(document) for document in self._documents]
        self._lengths = [len(document) for document in self._documents]
        self._avg_length = sum(self._lengths) / len(self._lengths)

        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            document_frequency.update(set(document))
        size = len(self.tools)
        self._idf = {
            term: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str, *, limit: int = 10) -> list[RankedTool]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_terms = tokenize(query)
        scored: list[tuple[str, float]] = []

        for tool, term_counts, document_length in zip(
            self.tools,
            self._term_counts,
            self._lengths,
        ):
            score = 0.0
            normalization = 1 - self.b + self.b * document_length / self._avg_length
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * normalization
                score += self._idf.get(term, 0.0) * numerator / denominator
            scored.append((tool.id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedTool(tool_id=tool_id, score=score, rank=index)
            for index, (tool_id, score) in enumerate(scored[:limit], start=1)
        ]
