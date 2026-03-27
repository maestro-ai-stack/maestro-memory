from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from maestro_memory.core.models import SearchResult
from maestro_memory.retrieval.bm25 import fts5_search_entities, fts5_search_facts
from maestro_memory.retrieval.embedding import cosine_top_k
from maestro_memory.retrieval.graph import graph_neighbors
from maestro_memory.retrieval.temporal import filter_temporal, temporal_score

if TYPE_CHECKING:
    from maestro_memory.core.store import Store
    from maestro_memory.retrieval.embedding import EmbeddingProvider


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
) -> list[SearchResult]:
    """Orchestrate BM25 + embedding + graph search, fuse with RRF, optionally rerank."""
    # When reranking, fetch more candidates for the reranker to work with
    reranker_available = rerank and _get_reranker() is not None
    fetch_limit = limit * (5 if reranker_available else 3)

    # 1. BM25 search
    bm25_results = await fts5_search_facts(store, query, limit=fetch_limit)

    # 2. Embedding search
    emb_results: list[tuple[int, float]] = []
    if embedding_provider:
        query_emb = await embedding_provider.embed(query)
        if query_emb is not None:
            cur = await store.db.execute("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")
            rows = await cur.fetchall()
            fact_embeddings = []
            for row in rows:
                emb = np.frombuffer(row[1], dtype=np.float32)
                fact_embeddings.append((row[0], emb))
            if fact_embeddings:
                emb_results = cosine_top_k(query_emb, fact_embeddings, k=fetch_limit)

    # 3. Graph expansion
    graph_results: list[tuple[int, float]] = []
    entity_hits = await fts5_search_entities(store, query, limit=5)
    if entity_hits:
        entity_ids = [eid for eid, _ in entity_hits]
        graph_results = await graph_neighbors(store, entity_ids, hops=2, current_only=current_only)

    # 4. RRF fusion
    sources_to_fuse = [r for r in [bm25_results, emb_results, graph_results] if r]
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
        final_score = rrf_score * t_score

        entity = None
        if fact.entity_id:
            entity = await store.get_entity(fact.entity_id)

        await store.increment_access(fact_id)

        results.append(SearchResult(fact=fact, score=final_score, source="fused", entity=entity))

        if len(results) >= candidate_limit:
            break

    # 6. Cross-encoder rerank (if available and enabled)
    if reranker_available and len(results) > limit:
        results = rerank_results(query, results, limit)
    else:
        results = results[:limit]

    return results
