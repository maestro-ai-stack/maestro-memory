from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


class EmbeddingProvider(abc.ABC):
    """Abstract embedding provider."""

    @abc.abstractmethod
    async def embed(self, text: str) -> ndarray | None:
        ...


class NullEmbeddingProvider(EmbeddingProvider):
    """Returns None when no provider is available."""

    async def embed(self, text: str) -> ndarray | None:  # noqa: ARG002
        return None


class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses sentence-transformers for local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    async def embed(self, text: str) -> ndarray | None:
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)


class BGEEmbeddingProvider(EmbeddingProvider):
    """BGE-M3 多语言嵌入，懒加载。"""

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-m3")

    async def embed(self, text: str) -> ndarray | None:
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)


# ── 工厂函数 ──────────────────────────────────────────────

_MODEL_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "bge-m3": BGEEmbeddingProvider,
    "all-MiniLM-L6-v2": LocalEmbeddingProvider,
}


def get_embedding_provider(provider: str = "local", model: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Factory: return the best available provider, gracefully falling back."""
    if provider != "local":
        return NullEmbeddingProvider()
    try:
        import sentence_transformers as _st  # noqa: F401
        _ = _st
    except ImportError:
        return NullEmbeddingProvider()
    cls = _MODEL_PROVIDERS.get(model)
    if cls is BGEEmbeddingProvider:
        return BGEEmbeddingProvider()
    return LocalEmbeddingProvider(model)


def cosine_similarity(a: ndarray, b: ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_top_k(
    query_emb: ndarray,
    fact_embeddings: list[tuple[int, ndarray]],
    k: int = 30,
) -> list[tuple[int, float]]:
    """Return top-k (fact_id, similarity) pairs by cosine similarity."""
    scored = []
    for fid, emb in fact_embeddings:
        sim = cosine_similarity(query_emb, emb)
        scored.append((fid, sim))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
