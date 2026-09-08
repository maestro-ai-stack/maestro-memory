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
    resp = await client.post("/search", json={"query": "dark mode", "limit": 5, "rerank": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "meta" in data
    assert isinstance(data["query_id"], int)
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


@pytest.mark.asyncio
async def test_add_idempotency_key_prevents_retry_duplicates(client):
    payload = {"content": "The grant deadline is 28 September", "idempotency_key": "gmail:message-1"}

    first = (await client.post("/add", json=payload)).json()
    second = (await client.post("/add", json=payload)).json()
    status = (await client.get("/status")).json()

    assert first["facts_added"] == 1
    assert first["idempotent_replay"] is False
    assert second["episode_id"] == first["episode_id"]
    assert second["facts_added"] == 0
    assert second["idempotent_replay"] is True
    assert status["episodes"] == 1
    assert status["facts"] == 1


@pytest.mark.asyncio
async def test_feedback_uses_exact_short_query_id(client):
    await client.post("/add", json={"content": "The client approved Project Atlas"})
    search = (await client.post("/search", json={"query": "Project Atlas"})).json()
    fact_id = next(item["fact_id"] for item in search["results"] if item["fact_id"] >= 0)

    feedback = await client.post(
        "/feedback", json={"query_id": search["query_id"], "used_fact_ids": [fact_id]}
    )

    assert feedback.status_code == 200
    assert feedback.json() == {"status": "ok", "facts_updated": 1}
