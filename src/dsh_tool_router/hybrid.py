from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import RankedTool
from .retrieval import ToolRetriever


class HybridRouter:
    """Fuse ranked lists with Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrievers: Sequence[ToolRetriever],
        *,
        rrf_k: float = 60.0,
        candidate_limit: int = 100,
    ) -> None:
        self.retrievers = tuple(retrievers)
        if len(self.retrievers) < 2:
            raise ValueError("hybrid fusion requires at least two retrievers")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        self.rrf_k = rrf_k
        self.candidate_limit = candidate_limit

    def rank(self, query: str, *, limit: int = 10) -> list[RankedTool]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        scores: dict[str, float] = defaultdict(float)
        for retriever in self.retrievers:
            for item in retriever.rank(query, limit=self.candidate_limit):
                scores[item.tool_id] += 1.0 / (self.rrf_k + item.rank)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            RankedTool(tool_id=tool_id, score=score, rank=index)
            for index, (tool_id, score) in enumerate(ranked[:limit], start=1)
        ]
