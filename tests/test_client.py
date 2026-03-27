"""Tests for the thin HTTP client."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from maestro_memory.server.app import create_app
from maestro_memory.server import lifecycle
from maestro_memory.client import MemoryClient


@pytest_asyncio.fixture
async def server_and_client(tmp_path):
    """Create a test server and MemoryClient backed by ASGITransport."""
    app = create_app(db_path=tmp_path / "test.db")
    async with lifecycle.lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = MemoryClient(base_url="http://test", http_client=http)
            yield client


@pytest.mark.asyncio
async def test_client_health(server_and_client):
    data = await server_and_client.health()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_client_add_and_search(server_and_client):
    client = server_and_client
    result = await client.add("User likes dark mode")
    assert result["facts_added"] >= 1

    results = await client.search("dark mode", rerank=False)
    assert len(results) >= 1
    assert "dark mode" in results[0]["content"]


@pytest.mark.asyncio
async def test_client_status(server_and_client):
    data = await server_and_client.status()
    assert "entities" in data
    assert "facts" in data
