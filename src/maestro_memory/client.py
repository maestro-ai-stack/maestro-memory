"""Thin HTTP client for mmem daemon."""
from __future__ import annotations

import httpx


class MemoryClient:
    """Thin HTTP client for mmem daemon."""

    def __init__(
        self,
        base_url: str = "http://localhost:19830",
        http_client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url
        self._http = http_client
        self._owns_http = http_client is None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self._base_url, timeout=30)
        return self._http

    async def search(
        self, query: str, limit: int = 10, rerank: bool = True, **kwargs
    ) -> list[dict]:
        http = await self._get_http()
        resp = await http.post(
            "/search", json={"query": query, "limit": limit, "rerank": rerank, **kwargs}
        )
        resp.raise_for_status()
        return resp.json()

    async def add(self, content: str, **kwargs) -> dict:
        http = await self._get_http()
        resp = await http.post("/add", json={"content": content, **kwargs})
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        http = await self._get_http()
        resp = await http.get("/status")
        resp.raise_for_status()
        return resp.json()

    async def feedback(self, query: str, used_fact_ids: list[int]) -> dict:
        http = await self._get_http()
        resp = await http.post("/feedback", json={"query": query, "used_fact_ids": used_fact_ids})
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict:
        http = await self._get_http()
        resp = await http.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None
