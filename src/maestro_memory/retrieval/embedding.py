from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger("mmem.embedding")

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
    """BGE-M3 multilingual embedding, lazy-loaded."""

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


# ── Factory ──────────────────────────────────────────────

_MODEL_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "bge-m3": BGEEmbeddingProvider,
    "all-MiniLM-L6-v2": LocalEmbeddingProvider,
}

# Short names accepted in config.toml, resolved to sentence-transformers ids.
# Any id sentence-transformers understands may also be given verbatim.
_MODEL_ALIASES: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "embeddinggemma": "google/embeddinggemma-300m",
}


def get_embedding_provider(provider: str = "local", model: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Factory: return the configured provider.

    Degradation to :class:`NullEmbeddingProvider` is announced loudly. A silent
    fallback here disables semantic recall entirely while search keeps returning
    plausible lexical results, so the failure is invisible — the mode that left
    the predecessor system without embeddings for two and a half months.
    """
    if provider != "local":
        logger.warning(
            "embedding provider %r is not supported; semantic recall is DISABLED "
            "and search will fall back to keyword matching only", provider,
        )
        return NullEmbeddingProvider()
    try:
        import sentence_transformers as _st  # noqa: F401
        _ = _st
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed; semantic recall is DISABLED "
            "and search will fall back to keyword matching only. "
            "Install it with: pip install 'maestro-memory[dense]'",
        )
        return NullEmbeddingProvider()
    resolved = _MODEL_ALIASES.get(model, model)
    if _MODEL_PROVIDERS.get(model) is BGEEmbeddingProvider:
        return BGEEmbeddingProvider()
    return LocalEmbeddingProvider(resolved)


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
