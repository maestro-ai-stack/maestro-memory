from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro_memory.core.store import Store


async def fts5_search_facts(store: Store, query: str, limit: int = 30) -> list[tuple[int, float]]:
    """Search facts via FTS5 BM25. Returns list of (fact_id, bm25_score)."""
    # Escape FTS5 special characters
    safe_query = _escape_fts(query)
    if not safe_query.strip():
        return []
    try:
        cur = await store.db.execute(
            "SELECT rowid, bm25(facts_fts) AS score FROM facts_fts WHERE facts_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (safe_query, limit),
        )
        rows = await cur.fetchall()
        # bm25() returns negative values (lower = better match), so negate for ranking
        return [(row[0], -row[1]) for row in rows]
    except Exception:
        return []


async def fts5_search_entities(store: Store, query: str, limit: int = 30) -> list[tuple[int, float]]:
    """Search entities via FTS5 BM25. Returns list of (entity_id, bm25_score)."""
    safe_query = _escape_fts(query)
    if not safe_query.strip():
        return []
    try:
        cur = await store.db.execute(
            "SELECT rowid, bm25(entities_fts) AS score FROM entities_fts WHERE entities_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (safe_query, limit),
        )
        rows = await cur.fetchall()
        return [(row[0], -row[1]) for row in rows]
    except Exception:
        return []


def _escape_fts(query: str) -> str:
    """Escape special FTS5 characters, keeping words as OR-joined tokens."""
    # Remove FTS5 operators, keep alphanumeric and spaces
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
    tokens = cleaned.split()
    if not tokens:
        return ""
    # Join with OR for broad matching
    return " OR ".join(tokens)
