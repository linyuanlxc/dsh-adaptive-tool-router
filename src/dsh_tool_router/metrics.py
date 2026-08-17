from __future__ import annotations

from collections.abc import Iterable, Sequence
import math


def _validate(labels: set[str], ranked_ids: Sequence[str], k: int) -> None:
    if not labels:
        raise ValueError("labels must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked ids must be unique")


def recall_at_k(labels: set[str], ranked_ids: Sequence[str], k: int) -> float:
    _validate(labels, ranked_ids, k)
    return len(labels.intersection(ranked_ids[:k])) / len(labels)


def reciprocal_rank(labels: set[str], ranked_ids: Sequence[str]) -> float:
    _validate(labels, ranked_ids, 1)
    for rank, tool_id in enumerate(ranked_ids, start=1):
        if tool_id in labels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: set[str], ranked_ids: Sequence[str], k: int) -> float:
    _validate(labels, ranked_ids, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, tool_id in enumerate(ranked_ids[:k], start=1)
        if tool_id in labels
    )
    ideal_hits = min(len(labels), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return sum(values) / len(values)


def evaluate_rankings(
    examples: Iterable[tuple[set[str], Sequence[str]]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, float]:
    examples = tuple(examples)
    if not examples:
        raise ValueError("at least one example is required")

    results = {
        "mrr": mean(reciprocal_rank(labels, ranked) for labels, ranked in examples)
    }
    for k in cutoffs:
        results[f"recall@{k}"] = mean(
            recall_at_k(labels, ranked, k) for labels, ranked in examples
        )
        results[f"ndcg@{k}"] = mean(
            ndcg_at_k(labels, ranked, k) for labels, ranked in examples
        )
    return results
