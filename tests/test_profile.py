"""Tests for UserProfile."""
from __future__ import annotations

import pytest
import numpy as np
from maestro_memory.core.profile import UserProfile


def test_entity_affinity():
    p = UserProfile()
    p.update_entity(entity_id=1, boost=1.0)
    p.update_entity(entity_id=1, boost=1.0)
    p.update_entity(entity_id=2, boost=1.0)
    assert p.get_affinity(1) > p.get_affinity(2)
    assert p.get_affinity(99) == 0.0


def test_decay():
    p = UserProfile()
    p.update_entity(entity_id=1, boost=1.0)
    initial = p.get_affinity(1)
    p.decay_all(factor=0.5)
    assert p.get_affinity(1) == pytest.approx(initial * 0.5)


def test_decay_prunes_small():
    p = UserProfile()
    p.update_entity(entity_id=1, boost=0.005)
    p.decay_all(factor=0.5)
    assert p.get_affinity(1) == 0.0  # pruned


def test_top_entities():
    p = UserProfile()
    for i in range(20):
        p.update_entity(i, boost=float(i))
    top = p.top_entities(5)
    assert len(top) == 5
    assert top[0][0] == 19  # highest boost


def test_topic_embedding():
    p = UserProfile()
    v1 = np.array([1.0, 0, 0, 0], dtype=np.float32)
    p.update_topic(v1)
    assert np.allclose(p.topic_embedding, v1)
    v2 = np.array([0, 1.0, 0, 0], dtype=np.float32)
    p.update_topic(v2, alpha=0.5)
    expected = 0.5 * v1 + 0.5 * v2
    assert np.allclose(p.topic_embedding, expected)
