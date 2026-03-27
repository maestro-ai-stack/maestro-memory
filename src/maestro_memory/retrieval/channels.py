"""Additional recall channels beyond BM25, embedding, and graph."""
from __future__ import annotations

from datetime import datetime, timedelta

from maestro_memory.core.profile import UserProfile
from maestro_memory.core.session import SessionState
from maestro_memory.core.store import Store


async def recall_user_interest(
    store: Store, profile: UserProfile, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 4: facts from high-affinity entities."""
    candidates: list[tuple[int, float]] = []
    for eid, affinity in profile.top_entities(10):
        cur = await store.db.execute(
            "SELECT id, importance FROM facts WHERE entity_id = ? AND valid_until IS NULL "
            "ORDER BY access_count DESC LIMIT 5",
            (eid,),
        )
        rows = await cur.fetchall()
        for row in rows:
            candidates.append((row[0], affinity * row[1]))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:limit]


async def recall_time_window(
    store: Store, days: int = 7, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 5: recent facts by creation time."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = await store.db.execute(
        "SELECT id, importance FROM facts WHERE created_at >= ? AND valid_until IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (cutoff, limit),
    )
    return [(row[0], row[1]) for row in await cur.fetchall()]


async def recall_session_context(
    store: Store, session: SessionState, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 6: facts from entities active in current session."""
    candidates: list[tuple[int, float]] = []
    for eid, activation in sorted(session.entity_activation.items(), key=lambda x: -x[1])[:10]:
        cur = await store.db.execute(
            "SELECT id, importance FROM facts WHERE entity_id = ? AND valid_until IS NULL "
            "ORDER BY access_count DESC LIMIT 5",
            (eid,),
        )
        rows = await cur.fetchall()
        for row in rows:
            if row[0] not in session.recent_fact_ids:
                candidates.append((row[0], activation * row[1]))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:limit]
