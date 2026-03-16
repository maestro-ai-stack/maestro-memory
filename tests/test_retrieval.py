"""Tests for retrieval components: BM25, temporal, fusion, embedding."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from maestro_memory.core.models import Fact
from maestro_memory.core.store import Store
from maestro_memory.retrieval.bm25 import fts5_search_facts
from maestro_memory.retrieval.embedding import (
    NullEmbeddingProvider,
    cosine_similarity,
)
from maestro_memory.retrieval.fusion import hybrid_search, reciprocal_rank_fusion
from maestro_memory.retrieval.temporal import filter_temporal, temporal_score


# ── BM25 ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bm25_search_facts(store: Store) -> None:
    await store.add_fact("Python is a great programming language")
    await store.add_fact("Rust is fast and safe")
    results = await fts5_search_facts(store, "Python programming")
    assert len(results) >= 1
    # First result should be the Python fact
    fact = await store.get_fact(results[0][0])
    assert fact is not None
    assert "Python" in fact.content


@pytest.mark.asyncio
async def test_bm25_no_results(store: Store) -> None:
    results = await fts5_search_facts(store, "")
    assert results == []


# ── Temporal ──────────────────────────────────────────────────────────────

def test_temporal_score_fresh() -> None:
    """A recently-created fact should score close to its importance."""
    now = datetime.now()
    fact = Fact(
        id=1, content="fresh", importance=0.8,
        valid_from=now.isoformat(), created_at=now.isoformat(),
    )
    score = temporal_score(fact, as_of=now)
    # exp(-0.01 * 0) = 1.0, so score ~ 0.8
    assert score == pytest.approx(0.8, abs=0.05)


def test_temporal_score_old() -> None:
    """A 100-day-old fact should score lower than a fresh one."""
    now = datetime.now()
    old = now - timedelta(days=100)
    fact = Fact(
        id=1, content="old", importance=0.8,
        valid_from=old.isoformat(), created_at=old.isoformat(),
    )
    score = temporal_score(fact, as_of=now)
    expected = 0.8 * math.exp(-0.01 * 100)
    assert score == pytest.approx(expected, abs=0.01)


def test_temporal_filter_current() -> None:
    """Expired facts are filtered out when current_only=True."""
    now = datetime.now()
    valid = Fact(id=1, content="valid", valid_from=now.isoformat(), valid_until=None)
    expired = Fact(id=2, content="expired", valid_from=now.isoformat(),
                   valid_until=(now - timedelta(days=1)).isoformat())
    result = filter_temporal([valid, expired], current_only=True)
    assert len(result) == 1
    assert result[0].id == 1


def test_temporal_filter_as_of() -> None:
    """Filter by as_of date excludes facts created after that date."""
    past = "2025-01-01T00:00:00"
    future_fact = Fact(id=1, content="future", valid_from="2026-06-01T00:00:00")
    past_fact = Fact(id=2, content="past", valid_from="2024-06-01T00:00:00")
    result = filter_temporal([future_fact, past_fact], current_only=False, as_of=past)
    assert len(result) == 1
    assert result[0].id == 2


# ── RRF Fusion ────────────────────────────────────────────────────────────

def test_rrf_fusion() -> None:
    list_a = [(10, 0.9), (20, 0.8)]
    list_b = [(20, 0.95), (30, 0.7)]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)
    ids = [item_id for item_id, _ in fused]
    # id=20 appears in both lists, so should rank highest
    assert ids[0] == 20


# ── Cosine Similarity ────────────────────────────────────────────────────

def test_cosine_similarity() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, c) == pytest.approx(0.0)


# ── NullEmbeddingProvider ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_null_embedding_provider() -> None:
    provider = NullEmbeddingProvider()
    result = await provider.embed("anything")
    assert result is None


# ── Hybrid Search (BM25 only, no embeddings) ────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_search_bm25_only(store: Store) -> None:
    await store.add_fact("Python is excellent for scripting")
    await store.add_fact("Rust is excellent for systems programming")
    results = await hybrid_search(store, "Python scripting", None, limit=5)
    assert len(results) >= 1
    assert "Python" in results[0].fact.content
