from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .models import RankedTool
from .retrieval import ToolRetriever


@dataclass(frozen=True)
class ToolSignal:
    global_uses: float = 0.0
    project_uses: float = 0.0
    successes: float = 0.0
    failures: float = 0.0
    average_latency_ms: float = 0.0


@dataclass(frozen=True)
class PersonalizationWeights:
    lexical: float = 0.65
    project: float = 0.15
    personal: float = 0.10
    reliability: float = 0.10
    latency_penalty: float = 0.05


class AdaptiveRouter:
    """Rerank a lexical candidate set with local project and outcome signals."""

    def __init__(
        self,
        base_router: ToolRetriever,
        *,
        weights: PersonalizationWeights | None = None,
        candidate_limit: int = 100,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        self.base_router = base_router
        self.weights = weights or PersonalizationWeights()
        self.candidate_limit = candidate_limit

    def rank(
        self,
        query: str,
        *,
        signals: Mapping[str, ToolSignal],
        limit: int = 10,
    ) -> list[RankedTool]:
        candidates = self.base_router.rank(query, limit=self.candidate_limit)
        if limit <= 0:
            raise ValueError("limit must be positive")

        max_lexical = max((candidate.score for candidate in candidates), default=0.0)
        max_global = max((signal.global_uses for signal in signals.values()), default=0.0)
        max_project = max((signal.project_uses for signal in signals.values()), default=0.0)
        max_latency = max(
            (signal.average_latency_ms for signal in signals.values()),
            default=0.0,
        )

        scored: list[tuple[str, float]] = []
        for candidate in candidates:
            signal = signals.get(candidate.tool_id, ToolSignal())
            lexical = candidate.score / max_lexical if max_lexical else 0.0
            project = _log_normalize(signal.project_uses, max_project)
            personal = _log_normalize(signal.global_uses, max_global)
            reliability = (signal.successes + 2) / (
                signal.successes + signal.failures + 4
            )
            latency = (
                signal.average_latency_ms / max_latency if max_latency else 0.0
            )

            score = (
                self.weights.lexical * lexical
                + self.weights.project * project
                + self.weights.personal * personal
                + self.weights.reliability * reliability
                - self.weights.latency_penalty * latency
            )
            scored.append((candidate.tool_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedTool(tool_id=tool_id, score=score, rank=rank)
            for rank, (tool_id, score) in enumerate(scored[:limit], start=1)
        ]


def _log_normalize(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return math.log1p(value) / math.log1p(maximum)
