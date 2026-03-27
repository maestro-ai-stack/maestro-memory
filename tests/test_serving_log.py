"""Tests for Phase 4: serving logs, feedback, and Thompson Sampling blender."""
from __future__ import annotations

import json

import numpy as np
import pytest
import pytest_asyncio

from maestro_memory.logging.serving_log import ServingLogger
from maestro_memory.ranking.blender import ThompsonBlender


# ── ServingLogger ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_search_writes_to_db(store):
    """log_search context manager inserts a row with query and latency."""
    logger = ServingLogger(store)

    async with logger.log_search("climate panel data") as entry:
        entry["candidate_ids"] = [1, 2, 3]
        entry["returned_ids"] = [1, 2]

    cur = await store.db.execute("SELECT * FROM serving_logs")
    rows = await cur.fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "climate panel data"
    assert json.loads(row["candidate_fact_ids"]) == [1, 2, 3]
    assert json.loads(row["returned_fact_ids"]) == [1, 2]
    assert row["latency_ms"] >= 0
    assert row["used_fact_ids"] is None  # no feedback yet


@pytest.mark.asyncio
async def test_record_feedback_updates_log(store):
    """record_feedback writes used_fact_ids to the most recent matching log."""
    logger = ServingLogger(store)

    # Insert two logs for the same query
    async with logger.log_search("era5 temperature") as entry:
        entry["returned_ids"] = [10, 20, 30]

    async with logger.log_search("era5 temperature") as entry:
        entry["returned_ids"] = [10, 20, 30, 40]

    # Feedback should update only the most recent entry
    await logger.record_feedback("era5 temperature", [10, 30])

    cur = await store.db.execute(
        "SELECT used_fact_ids FROM serving_logs ORDER BY id"
    )
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert rows[0]["used_fact_ids"] is None  # older entry untouched
    assert json.loads(rows[1]["used_fact_ids"]) == [10, 30]


@pytest.mark.asyncio
async def test_get_training_data(store):
    """get_training_data returns only logs that have feedback."""
    logger = ServingLogger(store)

    async with logger.log_search("query_a") as entry:
        entry["returned_ids"] = [1, 2]
    async with logger.log_search("query_b") as entry:
        entry["returned_ids"] = [3, 4]

    # Only give feedback for query_a
    await logger.record_feedback("query_a", [1])

    data = await logger.get_training_data(limit=100)
    assert len(data) == 1
    assert data[0]["query"] == "query_a"
    assert data[0]["used_fact_ids"] == [1]
    assert data[0]["returned_fact_ids"] == [1, 2]


# ── ThompsonBlender ────────────────────────────────────────────────


def test_sample_weights_shape_and_sum():
    """sample_weights returns an array of length n_channels that sums to 1."""
    blender = ThompsonBlender(n_channels=6)
    weights = blender.sample_weights()
    assert weights.shape == (6,)
    assert np.isclose(weights.sum(), 1.0)


def test_sample_weights_all_positive():
    """All sampled weights are non-negative (Beta distribution is >= 0)."""
    blender = ThompsonBlender(n_channels=6)
    for _ in range(50):
        weights = blender.sample_weights()
        assert (weights >= 0).all()


def test_update_increments_alpha_on_reward():
    """update with reward > 0.5 increments alpha."""
    blender = ThompsonBlender(n_channels=6)
    old_alpha = blender.alpha[0]
    old_beta = blender.beta[0]

    blender.update(0, reward=0.8)

    assert blender.alpha[0] == old_alpha + 1
    assert blender.beta[0] == old_beta  # unchanged


def test_update_increments_beta_on_low_reward():
    """update with reward <= 0.5 increments beta."""
    blender = ThompsonBlender(n_channels=6)
    old_alpha = blender.alpha[2]
    old_beta = blender.beta[2]

    blender.update(2, reward=0.3)

    assert blender.alpha[2] == old_alpha  # unchanged
    assert blender.beta[2] == old_beta + 1


def test_update_tracks_count():
    """n_updates increments on each update call."""
    blender = ThompsonBlender(n_channels=6)
    assert blender.n_updates == 0
    blender.update(0, 0.9)
    blender.update(1, 0.1)
    assert blender.n_updates == 2


def test_get_stats_structure():
    """get_stats returns correct keys and types."""
    blender = ThompsonBlender(n_channels=6)
    blender.update(0, 0.9)  # alpha[0] += 1
    stats = blender.get_stats()

    assert len(stats) == 6
    assert set(stats.keys()) == set(ThompsonBlender.CHANNEL_NAMES)

    bm25 = stats["bm25"]
    assert "mean" in bm25
    assert "alpha" in bm25
    assert "beta" in bm25
    assert bm25["alpha"] == 2.0  # initial 1 + 1 update
    assert bm25["beta"] == 1.0
    assert np.isclose(bm25["mean"], 2.0 / 3.0)
