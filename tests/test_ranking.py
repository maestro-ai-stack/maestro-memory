"""Tests for the ranking module."""
from __future__ import annotations

import numpy as np
import pytest
from maestro_memory.ranking.features import extract_features, FEATURE_NAMES
from maestro_memory.ranking.prerank import PreRanker
from maestro_memory.ranking.online import OnlineRanker


def test_extract_features_shape():
    feats = extract_features(
        query="dark mode preferences",
        fact_content="User prefers dark mode in all editors",
        fact_importance=0.7,
        fact_access_count=3,
        fact_created_at="2026-03-01T12:00:00",
        fact_last_accessed="2026-03-25T12:00:00",
        fact_entity_id=1,
        bm25_score=2.5,
        embed_score=0.85,
        graph_distance=1.0,
        entity_affinity=0.6,
        session_boost=0.3,
    )
    assert feats.shape == (12,)
    assert feats.dtype == np.float32
    assert len(FEATURE_NAMES) == 12


def test_extract_features_values():
    feats = extract_features(
        query="dark mode",
        fact_content="User prefers dark mode in all editors",
        fact_importance=0.7,
        fact_access_count=3,
        fact_created_at="2026-03-01T12:00:00",
        fact_last_accessed=None,
        fact_entity_id=None,
        bm25_score=2.5,
        embed_score=0.85,
        graph_distance=0.0,
    )
    assert feats[0] == pytest.approx(2.5)   # bm25
    assert feats[1] == pytest.approx(0.85)  # embed
    assert feats[4] == pytest.approx(0.7)   # importance
    assert feats[5] == pytest.approx(3.0)   # access_count


def test_preranker_fallback():
    """Without a model, PreRanker falls back to BM25 score."""
    pr = PreRanker()
    assert not pr.is_loaded
    features = np.array([
        [2.5, 0.8, 1.0, 0.3, 0.7, 3, 40, 10, 5, 1.2, 0.5, 0.1],
        [1.0, 0.9, 0.0, 0.5, 0.9, 5, 30, 2, 1, 2.0, 0.8, 0.3],
    ], dtype=np.float32)
    scores = pr.predict(features)
    assert scores[0] == pytest.approx(2.5)  # falls back to bm25
    assert scores[1] == pytest.approx(1.0)


def test_preranker_rank():
    pr = PreRanker()
    features = np.array([
        [2.5, 0.8, 1.0, 0.3, 0.7, 3, 40, 10, 5, 1.2, 0.5, 0.1],
        [3.0, 0.9, 0.0, 0.5, 0.9, 5, 30, 2, 1, 2.0, 0.8, 0.3],
        [1.0, 0.5, 2.0, 0.1, 0.3, 1, 20, 30, 20, 0.5, 0.1, 0.0],
    ], dtype=np.float32)
    ranked = pr.rank(features, [10, 20, 30], limit=2)
    assert len(ranked) == 2
    assert ranked[0][0] == 20  # highest bm25 = 3.0
    assert ranked[1][0] == 10  # second highest = 2.5


def test_online_ranker_predict_default():
    ranker = OnlineRanker()
    feats = {"bm25_score": 2.0, "embed_score": 0.8}
    score = ranker.predict(feats)
    assert 0.0 <= score <= 1.0


def test_online_ranker_learn():
    try:
        import river  # noqa: F401
    except ImportError:
        pytest.skip("river not installed")
    ranker = OnlineRanker()
    for _ in range(10):
        ranker.update({"bm25_score": 3.0, "embed_score": 0.9}, used=True)
        ranker.update({"bm25_score": 0.1, "embed_score": 0.1}, used=False)
    assert ranker.n_updates == 20
    # After training, high scores should predict higher probability
    high = ranker.predict({"bm25_score": 3.0, "embed_score": 0.9})
    low = ranker.predict({"bm25_score": 0.1, "embed_score": 0.1})
    assert high > low
