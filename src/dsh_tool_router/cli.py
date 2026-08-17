from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dense import DenseRouter, SentenceTransformerEmbedder
from .hybrid import HybridRouter
from .io import load_queries, load_tools
from .metrics import evaluate_rankings
from .retrieval import BM25Router, ToolRetriever
from .toolret import load_toolret


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tool retrieval baseline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toolret-task", help="Official ToolRet task/config name")
    source.add_argument("--tools", help="Tool catalog JSONL")
    parser.add_argument("--queries", help="Query examples JSONL (required with --tools)")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval baseline to evaluate",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/e5-small-v2",
        help="Sentence-Transformers model for dense/hybrid retrieval",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/dsh-tool-router",
        help="Directory for cached tool embeddings",
    )
    return parser


def build_router(
    retriever: str,
    tools,
    *,
    embedding_model: str,
    cache_dir: str,
) -> ToolRetriever:
    if retriever == "bm25":
        return BM25Router(tools)

    embedder = SentenceTransformerEmbedder(embedding_model)
    cache_path = Path(cache_dir) / _safe_name(embedder.name) / "manifest.json"
    dense = DenseRouter(tools, embedder, cache_path=cache_path)
    if retriever == "dense":
        return dense
    return HybridRouter([BM25Router(tools), dense])


def main() -> None:
    args = build_parser().parse_args()
    if args.toolret_task:
        tools, examples = load_toolret(args.toolret_task)
        source = f"toolret:{args.toolret_task}"
    else:
        if not args.queries:
            raise SystemExit("--queries is required when --tools is used")
        tools = load_tools(args.tools)
        examples = load_queries(args.queries)
        source = "jsonl"
    max_k = max(args.k)
    router = build_router(
        args.retriever,
        tools,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
    )

    rankings = []
    for example in examples:
        ranked = router.rank(example.search_text, limit=max_k)
        rankings.append((set(example.labels), [item.tool_id for item in ranked]))

    result = {
        "model": args.retriever,
        "source": source,
        "tool_count": len(tools),
        "query_count": len(examples),
        "metrics": evaluate_rankings(rankings, cutoffs=args.k),
    }
    if args.retriever in {"dense", "hybrid"}:
        result["embedding_model"] = args.embedding_model
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    main()
