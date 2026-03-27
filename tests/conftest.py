"""Shared fixtures for maestro-memory tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from maestro_memory.core.memory import Memory
from maestro_memory.core.store import Store


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    """Provide an initialised Store backed by a temp DB."""
    s = Store(tmp_path / "test.db")
    await s.init()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def memory(tmp_path: Path):
    """Provide an initialised Memory backed by a temp DB."""
    m = Memory(path=tmp_path / "mem.db")
    await m.init()
    yield m
    await m.close()
