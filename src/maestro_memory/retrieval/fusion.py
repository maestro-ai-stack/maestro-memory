from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from maestro_memory.core.models import SearchResult
from maestro_memory.retrieval.ann_index import ANNIndex
from maestro_memory.retrieval.bm25 import fts5_search_entities, fts5_search_facts
from maestro_memory.retrieval.channels import recall_session_context, recall_time_window, recall_user_interest
from maestro_memory.retrieval.embedding import cosine_top_k
from maestro_memory.retrieval.graph import graph_neighbors
from maestro_memory.retrieval.query_expansion import expand_query
from maestro_memory.retrieval.temporal import filter_temporal, temporal_score

if TYPE_CHECKING:
    from maestro_memory.core.profile import UserProfile
    from maestro_memory.core.session import SessionState
    from maestro_memory.core.store import Store
    from maestro_memory.retrieval.embedding import EmbeddingProvider


# ── ANN index (module-level singleton) ────────────────────────

_ann_index: ANNIndex | None = None


def get_ann_index() -> ANNIndex | None:
    """Return the module-level ANN index, or None if not set."""
    return _ann_index


def set_ann_index(index: ANNIndex | None) -> None:
    """Set the module-level ANN index."""
    global _ann_index
    _ann_index = index


# ── Cross-encoder reranker (lazy-loaded) ──────────────────────

_reranker = None
_reranker_failed = False


def _get_reranker():
    """Lazy-load cross-encoder reranker. Returns None if unavailable."""
    global _reranker, _reranker_failed
    if _reranker is not None:
        return _reranker
    if _reranker_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _reranker
    except Exception:
        _reranker_failed = True
        return None


def rerank_results(query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Rerank results using cross-encoder. Falls back to original order if unavailable."""
    reranker = _get_reranker()
    if reranker is None or len(results) <= limit:
        return results[:limit]
    pairs = [(query, r.fact.content[:500]) for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: -x[1])
    return [r for r, _ in ranked[:limit]]


# ── RRF fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(*result_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[int, float] = defaultdict(float)
    for results in result_lists:
        for rank, (item_id, _) in enumerate(results):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── Main search pipeline ─────────────────────────────────────

async def hybrid_search(
    store: Store,
    query: str,
    embedding_provider: EmbeddingProvider | None,
    *,
    limit: int = 10,
    current_only: bool = True,
    as_of: str | None = None,
    rerank: bool = True,
    profile: UserProfile | None = None,
    session: SessionState | None = None,
    ann_index: ANNIndex | None = None,
    min_score: float = 0.0,
    diverse: bool = False,
) -> list[SearchResult]:
    """Orchestrate 6-channel search (BM25 + embedding + graph + user interest + time + session), fuse with RRF, optionally rerank."""
    # When reranking, fetch more candidates for the reranker to work with
    reranker_available = rerank and _get_reranker() is not None
    fetch_limit = limit * (5 if reranker_available else 3)

    # Multi-query expansion: generate 1-4 variant queries
    queries = expand_query(query)

    # 1. BM25 search (union across all query variants)
    bm25_results: list[tuple[int, float]] = []
    bm25_seen: set[int] = set()
    for q in queries:
        hits = await fts5_search_facts(store, q, limit=fetch_limit)
        for fid, score in hits:
            if fid not in bm25_seen:
                bm25_seen.add(fid)
                bm25_results.append((fid, score))

    # 2. Embedding search (union across all query variants)
    emb_results: list[tuple[int, float]] = []
    if embedding_provider:
        emb_seen: set[int] = set()
        for q in queries:
            query_emb = await embedding_provider.embed(q)
            if query_emb is None:
                continue
            ann = ann_index or get_ann_index()
            if ann is not None and ann.size > 0:
                hits = ann.search(query_emb, k=fetch_limit)
            else:
                cur = await store.db.execute("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")
                rows = await cur.fetchall()
                fact_embeddings = [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]
                hits = cosine_top_k(query_emb, fact_embeddings, k=fetch_limit) if fact_embeddings else []
            for fid, score in hits:
                if fid not in emb_seen:
                    emb_seen.add(fid)
                    emb_results.append((fid, score))

    # 3. Graph expansion
    graph_results: list[tuple[int, float]] = []
    entity_hits = await fts5_search_entities(store, query, limit=5)
    if entity_hits:
        entity_ids = [eid for eid, _ in entity_hits]
        graph_results = await graph_neighbors(store, entity_ids, hops=2, current_only=current_only)

    # 4. User interest channel
    interest_results: list[tuple[int, float]] = []
    if profile and profile.entity_affinity:
        interest_results = await recall_user_interest(store, profile, limit=fetch_limit)

    # 5. Time window channel
    time_results = await recall_time_window(store, days=7, limit=fetch_limit)

    # 6. Session context channel
    session_results: list[tuple[int, float]] = []
    if session and session.entity_activation:
        session_results = await recall_session_context(store, session, limit=fetch_limit)

    # RRF fusion
    sources_to_fuse = [r for r in [bm25_results, emb_results, graph_results,
                                    interest_results, time_results, session_results] if r]
    if not sources_to_fuse:
        return []
    fused = reciprocal_rank_fusion(*sources_to_fuse)

    # 5. Load facts, filter, score — collect more candidates when reranking
    candidate_limit = fetch_limit if reranker_available else limit
    results: list[SearchResult] = []
    for fact_id, rrf_score in fused:
        fact = await store.get_fact(fact_id)
        if not fact:
            continue

        filtered = filter_temporal([fact], current_only, as_of)
        if not filtered:
            continue

        as_of_dt = datetime.fromisoformat(as_of) if as_of else None
        t_score = temporal_score(fact, as_of=as_of_dt)
        # Importance boosting: facts with high importance get multiplicative boost
        importance_boost = 1.0 + fact.importance * 2  # importance 0.9 → 2.8x boost
        final_score = rrf_score * t_score * importance_boost

        entity = None
        if fact.entity_id:
            entity = await store.get_entity(fact.entity_id)

        results.append(SearchResult(fact=fact, score=final_score, source="fused", entity=entity))

        if len(results) >= candidate_limit:
            break

    # 6. Cross-encoder rerank (if available and enabled)
    if reranker_available and len(results) > limit:
        results = rerank_results(query, results, limit)
    else:
        results = results[:limit]

    # 7. MMR diversity reranking (for aggregation queries)
    if diverse and embedding_provider and len(results) > 1:
        from maestro_memory.retrieval.mmr import mmr_rerank
        fact_embeddings: dict[int, np.ndarray] = {}
        for r in results:
            cur = await store.db.execute("SELECT embedding FROM facts WHERE id = ?", (r.fact.id,))
            row = await cur.fetchone()
            if row and row[0]:
                fact_embeddings[r.fact.id] = np.frombuffer(row[0], dtype=np.float32)
        query_emb = await embedding_provider.embed(query)
        results = mmr_rerank(results, fact_embeddings, query_emb, lambda_param=0.6, limit=limit)

    # 8. Confidence gate: filter results below min_score
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]

    return results
