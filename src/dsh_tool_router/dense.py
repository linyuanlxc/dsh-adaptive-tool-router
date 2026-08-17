from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Protocol

from .models import RankedTool, ToolDocument


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier used in cache keys and evaluation reports."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text."""


class DenseRouter:
    """Cosine-similarity dense retriever with optional on-disk embedding cache."""

    def __init__(
        self,
        tools: Sequence[ToolDocument],
        embedder: Embedder,
        *,
        cache_path: str | Path | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        self.tools = tuple(tools)
        if not self.tools:
            raise ValueError("at least one tool is required")
        if len({tool.id for tool in self.tools}) != len(self.tools):
            raise ValueError("tool ids must be unique")

        self.embedder = embedder
        self.query_prefix, self.document_prefix = _resolve_prefixes(
            embedder.name,
            query_prefix,
            document_prefix,
        )
        documents = [
            f"{self.document_prefix}{tool.retrieval_text()}" for tool in self.tools
        ]
        self._embeddings = _load_or_encode(
            tools=self.tools,
            documents=documents,
            embedder=embedder,
            cache_path=Path(cache_path) if cache_path is not None else None,
        )
        if len({len(vector) for vector in self._embeddings}) != 1:
            raise ValueError("all tool embeddings must have the same dimension")

    def rank(self, query: str, *, limit: int = 10) -> list[RankedTool]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_vector = _l2_normalize(
            self.embedder.embed([f"{self.query_prefix}{query}"])[0]
        )
        if len(query_vector) != len(self._embeddings[0]):
            raise ValueError("query embedding dimension does not match tool embeddings")

        scored = [
            (tool.id, _dot(query_vector, document_vector))
            for tool, document_vector in zip(self.tools, self._embeddings)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedTool(tool_id=tool_id, score=score, rank=index)
            for index, (tool_id, score) in enumerate(scored[:limit], start=1)
        ]


class SentenceTransformerEmbedder:
    """Optional Sentence-Transformers backend; imported only when used."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        *,
        batch_size: int = 64,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    'Dense retrieval requires: pip install -e ".[dense]"'
                ) from error
            self._model = SentenceTransformer(self.model_name)
        return self._model


def _resolve_prefixes(
    model_name: str,
    query_prefix: str | None,
    document_prefix: str | None,
) -> tuple[str, str]:
    defaults = _default_prefixes(model_name)
    return (
        defaults[0] if query_prefix is None else query_prefix,
        defaults[1] if document_prefix is None else document_prefix,
    )


def _default_prefixes(model_name: str) -> tuple[str, str]:
    lowered = model_name.lower()
    if "e5-" in lowered:
        return "query: ", "passage: "
    if "bge-" in lowered:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


def _load_or_encode(
    *,
    tools: Sequence[ToolDocument],
    documents: Sequence[str],
    embedder: Embedder,
    cache_path: Path | None,
) -> list[list[float]]:
    fingerprints = [_fingerprint(document) for document in documents]
    cached = _read_cache(cache_path, embedder.name) if cache_path is not None else {}
    pending_indexes = [
        index
        for index, (tool, fingerprint) in enumerate(zip(tools, fingerprints))
        if cached.get(tool.id, {}).get("fingerprint") != fingerprint
    ]
    if pending_indexes:
        encoded = [
            _l2_normalize(vector)
            for vector in embedder.embed([documents[index] for index in pending_indexes])
        ]
        if len(encoded) != len(pending_indexes):
            raise ValueError("embedder must return one vector per document")
        for index, vector in zip(pending_indexes, encoded):
            cached[tools[index].id] = {
                "fingerprint": fingerprints[index],
                "vector": vector,
            }
        if cache_path is not None:
            _write_cache(cache_path, embedder.name, cached)

    embeddings = []
    for tool in tools:
        item = cached.get(tool.id)
        if item is None:
            raise ValueError(f"missing embedding for tool {tool.id!r}")
        embeddings.append(list(map(float, item["vector"])))
    return embeddings


def _read_cache(path: Path, model_name: str) -> dict[str, dict[str, object]]:
    manifest_path = path
    vectors_path = path.with_suffix(".bin")
    if not manifest_path.exists() or not vectors_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if manifest.get("model") != model_name:
        return {}
    ids = manifest.get("ids")
    fingerprints = manifest.get("fingerprints")
    dim = manifest.get("dim")
    if not isinstance(ids, list) or not isinstance(fingerprints, list):
        return {}
    if not isinstance(dim, int) or dim <= 0 or len(ids) != len(fingerprints):
        return {}
    try:
        vectors = _read_vectors(vectors_path, len(ids), dim)
    except (OSError, struct.error):
        return {}
    return {
        str(tool_id): {"fingerprint": str(fingerprint), "vector": vector}
        for tool_id, fingerprint, vector in zip(ids, fingerprints, vectors)
    }


def _write_cache(
    path: Path,
    model_name: str,
    items: dict[str, dict[str, object]],
) -> None:
    ids = sorted(items)
    vectors = [list(map(float, items[tool_id]["vector"])) for tool_id in ids]
    if not vectors:
        return
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("cached embeddings must have the same dimension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model_name,
                "dim": dim,
                "ids": ids,
                "fingerprints": [items[tool_id]["fingerprint"] for tool_id in ids],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_vectors(path.with_suffix(".bin"), vectors)


def _write_vectors(path: Path, vectors: Sequence[Sequence[float]]) -> None:
    dim = len(vectors[0])
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack(f"{dim}f", *vector))


def _read_vectors(path: Path, count: int, dim: int) -> list[list[float]]:
    payload = path.read_bytes()
    values = struct.unpack(f"{count * dim}f", payload)
    return [list(values[index * dim : (index + 1) * dim]) for index in range(count)]


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
