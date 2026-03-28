"""Tests for the FastAPI daemon server."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from maestro_memory.server.app import create_app
from maestro_memory.server import lifecycle


@pytest_asyncio.fixture
async def client(tmp_path):
    """Create an ASGI test client with lifespan managed."""
    app = create_app(db_path=tmp_path / "test.db")
    # Manually run lifespan since ASGITransport doesn't trigger it
    async with lifecycle.lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "entities" in data
    assert "facts" in data


@pytest.mark.asyncio
async def test_add_and_search(client):
    # Add a fact
    resp = await client.post("/add", json={"content": "User prefers dark mode in all editors"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode_id"] >= 1
    assert data["facts_added"] >= 1

    # Search for it
    resp = await client.post("/search", json={"query": "editor preferences", "limit": 5, "rerank": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "meta" in data
    results = data["results"]
    assert len(results) >= 1
    assert "dark mode" in results[0]["content"]
    # Verify meta structure
    meta = data["meta"]
    assert "confidence" in meta
    assert "best_score" in meta
    assert "suggestion" in meta


@pytest.mark.asyncio
async def test_status(client):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "facts" in data


@pytest.mark.asyncio
async def test_add_with_entity(client):
    resp = await client.post("/add", json={
        "content": "Prefers functional programming",
        "entity_name": "coding-style",
        "entity_type": "preference",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["entities_created"] == 1
