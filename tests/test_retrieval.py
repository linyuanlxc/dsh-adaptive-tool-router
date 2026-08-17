from dsh_tool_router.metrics import evaluate_rankings, ndcg_at_k, recall_at_k
from dsh_tool_router.models import ToolDocument
from dsh_tool_router.personalization import AdaptiveRouter, ToolSignal
from dsh_tool_router.retrieval import BM25Router


def test_bm25_ranks_relevant_tool_first() -> None:
    tools = [
        ToolDocument(
            id="weather",
            name="weather_forecast",
            description="Get a weather forecast for a city",
            parameters={"city": {"type": "string"}},
        ),
        ToolDocument(
            id="stocks",
            name="stock_price",
            description="Get the current stock market price",
            parameters={"ticker": {"type": "string"}},
        ),
    ]

    ranking = BM25Router(tools).rank("Will it rain in Beijing? weather forecast")

    assert ranking[0].tool_id == "weather"
    assert ranking[0].score > ranking[1].score


def test_multilabel_metrics() -> None:
    labels = {"search", "fetch"}
    ranked = ["search", "other", "fetch"]

    assert recall_at_k(labels, ranked, 1) == 0.5
    assert recall_at_k(labels, ranked, 3) == 1.0
    assert 0.0 < ndcg_at_k(labels, ranked, 3) < 1.0


def test_aggregate_metrics() -> None:
    result = evaluate_rankings(
        [
            ({"a"}, ["a", "b"]),
            ({"b"}, ["a", "b"]),
        ],
        cutoffs=[1, 2],
    )

    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["mrr"] == 0.75


def test_personalization_breaks_semantic_tie() -> None:
    tools = [
        ToolDocument(
            id="local",
            name="project_logs",
            description="Read task logs",
        ),
        ToolDocument(
            id="remote",
            name="remote_logs",
            description="Read task logs",
        ),
    ]
    router = AdaptiveRouter(BM25Router(tools))

    ranking = router.rank(
        "read task logs",
        signals={
            "local": ToolSignal(
                global_uses=20,
                project_uses=10,
                successes=9,
                failures=1,
            ),
            "remote": ToolSignal(
                global_uses=1,
                project_uses=0,
                successes=1,
                failures=2,
            ),
        },
    )

    assert ranking[0].tool_id == "local"
