from pathlib import Path

from dsh_tool_router.cli import build_parser
from dsh_tool_router.dense import DenseRouter
from dsh_tool_router.hybrid import HybridRouter
from dsh_tool_router.models import RankedTool, ToolDocument


class MappingEmbedder:
    def __init__(self, table: dict[str, list[float]], name: str = "fake") -> None:
        self.table = table
        self.name = name
        self.embedded: list[str] = []

    def embed(self, texts):
        encoded = []
        for text in texts:
            self.embedded.append(text)
            encoded.append(self.table[text])
        return encoded


class FixedRetriever:
    def __init__(self, tool_ids: list[str]) -> None:
        self.tool_ids = tool_ids

    def rank(self, query: str, *, limit: int = 10) -> list[RankedTool]:
        return [
            RankedTool(tool_id=tool_id, score=1.0, rank=rank)
            for rank, tool_id in enumerate(self.tool_ids[:limit], start=1)
        ]


def _tools() -> list[ToolDocument]:
    return [
        ToolDocument(
            id="weather",
            name="weather_forecast",
            description="Get a weather forecast",
        ),
        ToolDocument(
            id="stocks",
            name="stock_price",
            description="Get a stock price",
        ),
    ]


def test_dense_ranks_by_cosine_similarity() -> None:
    tools = _tools()
    embedder = MappingEmbedder(
        {
            f"passage: {tools[0].retrieval_text()}": [1.0, 0.0],
            f"passage: {tools[1].retrieval_text()}": [0.0, 1.0],
            "query: rain in Beijing": [0.9, 0.1],
        },
        name="intfloat/e5-small-v2",
    )

    ranking = DenseRouter(tools, embedder).rank("rain in Beijing")

    assert ranking[0].tool_id == "weather"
    assert ranking[0].score > ranking[1].score


def test_dense_cache_skips_unchanged_tools(tmp_path: Path) -> None:
    tools = _tools()
    table = {
        f"passage: {tools[0].retrieval_text()}": [1.0, 0.0],
        f"passage: {tools[1].retrieval_text()}": [0.0, 1.0],
        "query: weather": [1.0, 0.0],
    }
    first = MappingEmbedder(table, name="intfloat/e5-small-v2")
    cache_path = tmp_path / "manifest.json"
    DenseRouter(tools, first, cache_path=cache_path)

    second = MappingEmbedder(table, name="intfloat/e5-small-v2")
    router = DenseRouter(tools, second, cache_path=cache_path)
    ranking = router.rank("weather")

    assert second.embedded == ["query: weather"]
    assert ranking[0].tool_id == "weather"


def test_hybrid_rrf_promotes_consensus() -> None:
    ranking = HybridRouter(
        [
            FixedRetriever(["weather", "stocks", "search"]),
            FixedRetriever(["search", "weather", "stocks"]),
        ],
        rrf_k=60,
        candidate_limit=3,
    ).rank("anything", limit=3)

    assert [item.tool_id for item in ranking] == ["weather", "search", "stocks"]


def test_cli_accepts_dense_and_hybrid_flags() -> None:
    args = build_parser().parse_args(
        [
            "--tools",
            "examples/tools.jsonl",
            "--queries",
            "examples/queries.jsonl",
            "--retriever",
            "hybrid",
            "--embedding-model",
            "intfloat/e5-small-v2",
        ]
    )

    assert args.retriever == "hybrid"
    assert args.embedding_model == "intfloat/e5-small-v2"
