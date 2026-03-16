"""Tests for the Memory facade."""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro_memory.core.memory import Memory


@pytest.mark.asyncio
async def test_memory_init(memory: Memory) -> None:
    stats = await memory.status()
    assert stats["facts"] == 0
    assert stats["entities"] == 0


@pytest.mark.asyncio
async def test_add_and_search(memory: Memory) -> None:
    await memory.add("User prefers dark mode")
    results = await memory.search("dark mode")
    assert len(results) >= 1
    assert "dark mode" in results[0].fact.content


@pytest.mark.asyncio
async def test_add_multiple_search(memory: Memory) -> None:
    await memory.add("Python is preferred for scripts")
    await memory.add("TypeScript for frontend")
    await memory.add("Rust for performance-critical code")
    results = await memory.search("Python scripts", limit=3)
    assert len(results) >= 1
    # The Python fact should be the top hit
    assert "Python" in results[0].fact.content


@pytest.mark.asyncio
async def test_search_current_only(memory: Memory) -> None:
    result = await memory.add("Deprecated preference")
    # Invalidate the fact we just added
    facts = await memory.store.list_facts(current_only=False)
    assert len(facts) >= 1
    await memory.store.invalidate_fact(facts[0].id)

    # current_only search should not return it
    results = await memory.search("Deprecated", current_only=True)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_graph_entity(memory: Memory) -> None:
    e1, _ = await memory.store.get_or_create_entity("alice", "person")
    e2, _ = await memory.store.get_or_create_entity("bob", "person")
    await memory.store.add_relation(e1.id, "knows", e2.id)

    data = await memory.graph("alice")
    assert data["entity"] is not None
    assert data["entity"].name == "alice"
    assert len(data["relations"]) == 1
    assert len(data["neighbors"]) == 1
    assert data["neighbors"][0].name == "bob"


@pytest.mark.asyncio
async def test_graph_missing_entity(memory: Memory) -> None:
    data = await memory.graph("nonexistent")
    assert data["entity"] is None
    assert data["relations"] == []
    assert data["neighbors"] == []


@pytest.mark.asyncio
async def test_status(memory: Memory) -> None:
    stats = await memory.status()
    assert isinstance(stats, dict)
    assert "facts" in stats
    assert "entities" in stats
    assert "db_size_bytes" in stats


@pytest.mark.asyncio
async def test_memory_with_custom_path(tmp_path: Path) -> None:
    db_path = tmp_path / "custom" / "mem.db"
    mem = Memory(path=db_path)
    await mem.init()
    try:
        stats = await mem.status()
        assert stats["facts"] == 0
        assert db_path.exists()
    finally:
        await mem.close()
