"""12-dimensional feature extraction for ranking candidates."""
from __future__ import annotations
import math
import numpy as np
from datetime import datetime


def extract_features(
    query: str,
    fact_content: str,
    fact_importance: float,
    fact_access_count: int,
    fact_created_at: str,
    fact_last_accessed: str | None,
    fact_entity_id: int | None,
    bm25_score: float,
    embed_score: float,
    graph_distance: float,
    entity_affinity: float = 0.0,
    session_boost: float = 0.0,
) -> np.ndarray:
    """Extract 12-dim feature vector for a candidate fact."""
    now = datetime.now()

    try:
        created = datetime.fromisoformat(fact_created_at)
        days_since_created = max((now - created).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        days_since_created = 0

    try:
        if fact_last_accessed:
            accessed = datetime.fromisoformat(fact_last_accessed)
            days_since_accessed = max((now - accessed).total_seconds() / 86400, 0)
        else:
            days_since_accessed = days_since_created
    except (ValueError, TypeError):
        days_since_accessed = days_since_created

    query_words = set(query.lower().split())
    fact_words = set(fact_content.lower().split())
    entity_overlap = len(query_words & fact_words) / max(len(query_words), 1)

    base_level = math.log(fact_access_count + 1) - 0.5 * math.log(days_since_created + 1)

    return np.array([
        bm25_score,             # 0: BM25 relevance
        embed_score,            # 1: embedding cosine similarity
        graph_distance,         # 2: graph hop distance (0=direct)
        entity_overlap,         # 3: word overlap ratio
        fact_importance,        # 4: fact importance (0-1)
        fact_access_count,      # 5: historical access frequency
        len(fact_content),      # 6: fact text length
        days_since_created,     # 7: age of fact
        days_since_accessed,    # 8: recency of last access
        base_level,             # 9: ACT-R base-level activation
        entity_affinity,        # 10: user entity affinity
        session_boost,          # 11: session entity activation
    ], dtype=np.float32)


FEATURE_NAMES = [
    "bm25_score", "embed_score", "graph_distance", "entity_overlap",
    "importance", "access_count", "fact_length", "days_since_created",
    "days_since_accessed", "act_r_activation", "entity_affinity", "session_boost",
]
