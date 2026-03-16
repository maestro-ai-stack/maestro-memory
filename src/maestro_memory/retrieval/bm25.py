from __future__ import annotations

from typing import TYPE_CHECKING

from maestro_memory.retrieval.tokenizer import segment

if TYPE_CHECKING:
    from maestro_memory.core.store import Store


async def fts5_search_facts(store: Store, query: str, limit: int = 30) -> list[tuple[int, float]]:
    """Search facts via FTS5 BM25. Returns list of (fact_id, bm25_score)."""
    # 中文分词 + 转义 FTS5 特殊字符
    safe_query = _escape_fts(segment(query))
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
    safe_query = _escape_fts(segment(query))
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
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
    tokens = cleaned.split()
    if not tokens:
        return ""
    # 查询扩展：补充同义词/缩写/上下位词
    expanded = set(tokens)
    for t in tokens:
        for syn in _SYNONYMS.get(t.lower(), []):
            expanded.add(syn)
    return " OR ".join(expanded)


# ── 同义词表（轻量级查询扩展）────────────────────────────────
_SYNONYMS: dict[str, list[str]] = {
    # 医疗缩写
    "bp": ["blood", "pressure", "hypertension"],
    "blood": ["bp", "pressure"],
    "pressure": ["bp", "blood"],
    "hypertension": ["bp", "blood", "pressure"],
    "hr": ["heart", "rate"],
    # 食物/过敏
    "food": ["dietary", "allergy", "allergic", "vegetarian", "shellfish", "meal"],
    "restrictions": ["allergy", "allergic", "dietary", "preference"],
    "diet": ["food", "vegetarian", "allergic", "meal"],
    "allergy": ["allergic", "food", "restrictions"],
    # 症状
    "symptoms": ["complaint", "symptom", "reports", "dizziness", "headache", "pain"],
    "side": ["effect", "adverse", "reaction"],
    "effect": ["side", "adverse"],
    # 通用业务
    "status": ["update", "current", "progress"],
    "deadline": ["eta", "due", "date"],
    "eta": ["deadline", "estimate", "expected"],
    "blocker": ["blocked", "issue", "problem"],
}
