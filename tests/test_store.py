"""Tests for the SQLite storage layer."""
from __future__ import annotations

import pytest

from maestro_memory.core.store import Store


# ── Schema ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_creates_tables(store: Store) -> None:
    """Verify all expected tables exist after init."""
    cur = await store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = await cur.fetchall()
    names = {r[0] for r in rows}
    for expected in ("episodes", "entities", "relations", "facts"):
        assert expected in names, f"Missing table: {expected}"


# ── Episodes ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_get_episode(store: Store) -> None:
    eid = await store.add_episode("hello world", "manual", "ref-1")
    assert eid >= 1
    ep = await store.get_episode(eid)
    assert ep is not None
    assert ep.content == "hello world"
    assert ep.source_type == "manual"
    assert ep.source_ref == "ref-1"


# ── Entities ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_get_entity(store: Store) -> None:
    eid = await store.add_entity("python", "tool", "A programming language")
    entity = await store.get_entity(eid)
    assert entity is not None
    assert entity.name == "python"
    assert entity.entity_type == "tool"


@pytest.mark.asyncio
async def test_get_or_create_entity(store: Store) -> None:
    # First call creates
    entity1, created1 = await store.get_or_create_entity("rust", "tool")
    assert created1 is True
    assert entity1.name == "rust"

    # Second call returns existing
    entity2, created2 = await store.get_or_create_entity("rust", "tool")
    assert created2 is False
    assert entity2.id == entity1.id


# ── Facts ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_get_fact(store: Store) -> None:
    fid = await store.add_fact("Python is great", "preference", 0.8)
    fact = await store.get_fact(fid)
    assert fact is not None
    assert fact.content == "Python is great"
    assert fact.fact_type == "preference"
    assert fact.importance == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_invalidate_fact(store: Store) -> None:
    fid = await store.add_fact("old fact")
    await store.invalidate_fact(fid)
    fact = await store.get_fact(fid)
    assert fact is not None
    assert fact.valid_until is not None


# ── Relations ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_relation(store: Store) -> None:
    e1 = await store.add_entity("user")
    e2 = await store.add_entity("python")
    rid = await store.add_relation(e1, "uses", e2, confidence=0.9)
    assert rid >= 1


@pytest.mark.asyncio
async def test_get_relations_for_entity(store: Store) -> None:
    e1 = await store.add_entity("alice")
    e2 = await store.add_entity("bob")
    await store.add_relation(e1, "knows", e2)

    rels = await store.get_relations_for_entity(e1)
    assert len(rels) == 1
    assert rels[0].predicate == "knows"

    # Also visible from the other side
    rels2 = await store.get_relations_for_entity(e2)
    assert len(rels2) == 1


# ── List / Stats / Access ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_entities(store: Store) -> None:
    await store.add_entity("a")
    await store.add_entity("b")
    entities = await store.list_entities()
    assert len(entities) == 2


@pytest.mark.asyncio
async def test_list_facts(store: Store) -> None:
    await store.add_fact("fact one")
    await store.add_fact("fact two")
    facts = await store.list_facts()
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_fts_sync(store: Store) -> None:
    """BM25 search finds facts after add."""
    await store.add_fact("The quick brown fox jumps over the lazy dog")
    cur = await store.db.execute(
        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'fox'"
    )
    rows = await cur.fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_get_stats(store: Store) -> None:
    await store.add_episode("ep", "manual")
    await store.add_entity("e1")
    await store.add_fact("f1")
    stats = await store.get_stats()
    assert stats["episodes"] == 1
    assert stats["entities"] == 1
    assert stats["facts"] == 1
    assert stats["relations"] == 0
    assert "db_size_bytes" in stats


@pytest.mark.asyncio
async def test_increment_access(store: Store) -> None:
    fid = await store.add_fact("counter fact")
    await store.increment_access(fid)
    await store.increment_access(fid)
    fact = await store.get_fact(fid)
    assert fact is not None
    assert fact.access_count == 2
    assert fact.last_accessed is not None
