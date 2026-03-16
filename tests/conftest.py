"""Shared fixtures for maestro-memory tests."""
from __future__ import annotations

import pytest
from pathlib import Path

from maestro_memory.core.store import Store
from maestro_memory.core.memory import Memory


@pytest.fixture
async def store(tmp_path: Path):
    """Provide an initialised Store backed by a temp DB."""
    s = Store(tmp_path / "test.db")
    await s.init()
    yield s
    await s.close()


@pytest.fixture
async def memory(tmp_path: Path):
    """Provide an initialised Memory backed by a temp DB."""
    m = Memory(path=tmp_path / "mem.db")
    await m.init()
    yield m
    await m.close()
