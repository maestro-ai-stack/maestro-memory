"""Serving log — records every search for training data and analytics."""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from maestro_memory.core.store import Store


class ServingLogger:
    """Logs search requests for offline training and online learning."""

    def __init__(self, store: Store):
        self._store = store

    @asynccontextmanager
    async def log_search(self, query: str):
        """Context manager that times a search and logs results."""
        entry = {"query": query, "t0": time.perf_counter()}
        yield entry
        latency_ms = (time.perf_counter() - entry["t0"]) * 1000
        await self._store.db.execute(
            "INSERT INTO serving_logs (query, candidate_fact_ids, returned_fact_ids, latency_ms) VALUES (?, ?, ?, ?)",
            (
                query,
                json.dumps(entry.get("candidate_ids", [])),
                json.dumps(entry.get("returned_ids", [])),
                latency_ms,
            ),
        )
        await self._store.db.commit()

    async def record_feedback(self, query: str, used_fact_ids: list[int]) -> None:
        """Record which facts were actually used (implicit feedback)."""
        # SQLite doesn't support ORDER BY in UPDATE; use subquery for most recent log
        await self._store.db.execute(
            "UPDATE serving_logs SET used_fact_ids = ? WHERE id = (SELECT id FROM serving_logs WHERE query = ? ORDER BY id DESC LIMIT 1)",
            (json.dumps(used_fact_ids), query),
        )
        await self._store.db.commit()

    async def get_training_data(self, limit: int = 1000) -> list[dict]:
        """Get logs with feedback for training."""
        cur = await self._store.db.execute(
            "SELECT query, returned_fact_ids, used_fact_ids, features_json FROM serving_logs WHERE used_fact_ids IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [
            {
                "query": row[0],
                "returned_fact_ids": json.loads(row[1]) if row[1] else [],
                "used_fact_ids": json.loads(row[2]) if row[2] else [],
                "features": json.loads(row[3]) if row[3] else None,
            }
            for row in rows
        ]
