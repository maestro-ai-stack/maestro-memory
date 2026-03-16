"""Tests for ingestion fallback."""
from __future__ import annotations

import pytest

from maestro_memory.ingestion.fallback import fallback_extract


@pytest.mark.asyncio
async def test_fallback_extract() -> None:
    """Fallback returns a single ADD operation."""
    ops = await fallback_extract("User prefers dark mode")
    assert len(ops) == 1
    assert ops[0]["op"] == "ADD"


@pytest.mark.asyncio
async def test_fallback_extract_content() -> None:
    """Fallback preserves original content."""
    content = "The quick brown fox"
    ops = await fallback_extract(content)
    assert ops[0]["fact"] == content
