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


def reciprocal_rank_fusion(*result_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[int, float] = defaultdict(float)
    for results in result_lists:
        for rank, (item_id, _) in enumerate(results):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


async def hybrid_search(
    store: Store,
    query: str,
    embedding_provider: EmbeddingProvider | None,
    *,
    limit: int = 10,
    current_only: bool = True,
    as_of: str | None = None,
) -> list[SearchResult]:
    """Orchestrate BM25 + embedding + graph search, fuse with RRF, apply temporal decay."""
    fetch_limit = limit * 3

    # 1. BM25 search
    bm25_results = await fts5_search_facts(store, query, limit=fetch_limit)

    # 2. Embedding search
    emb_results: list[tuple[int, float]] = []
    if embedding_provider:
        query_emb = await embedding_provider.embed(query)
        if query_emb is not None:
            # Load all fact embeddings from DB
            cur = await store.db.execute("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")
            rows = await cur.fetchall()
            fact_embeddings = []
            for row in rows:
                emb = np.frombuffer(row[1], dtype=np.float32)
                fact_embeddings.append((row[0], emb))
            if fact_embeddings:
                emb_results = cosine_top_k(query_emb, fact_embeddings, k=fetch_limit)

    # 3. Graph expansion: find entities matching query, then expand
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

    # 5. Load facts, filter, score
    results: list[SearchResult] = []
    for fact_id, rrf_score in fused:
        fact = await store.get_fact(fact_id)
        if not fact:
            continue

        # Temporal filtering
        filtered = filter_temporal([fact], current_only, as_of)
        if not filtered:
            continue

        # Temporal decay scoring
        as_of_dt = datetime.fromisoformat(as_of) if as_of else None
        t_score = temporal_score(fact, as_of=as_of_dt)
        final_score = rrf_score * t_score

        # Load entity if linked
        entity = None
        if fact.entity_id:
            entity = await store.get_entity(fact.entity_id)

        # Increment access count
        await store.increment_access(fact_id)

        results.append(SearchResult(fact=fact, score=final_score, source="fused", entity=entity))

        if len(results) >= limit:
            break

    return results
