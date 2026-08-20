from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from maestro_memory.core.models import Fact, SearchResult
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
    from maestro_memory.ranking.blender import ThompsonBlender
    from maestro_memory.ranking.online import OnlineRanker
    from maestro_memory.ranking.prerank import PreRanker
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

def reciprocal_rank_fusion(
    *result_lists: list[tuple[int, float]],
    k: int = 60,
    weights: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    When *weights* is provided, each channel's RRF contribution is scaled
    by its weight.  Otherwise all channels contribute equally (original behavior).
    """
    scores: dict[int, float] = defaultdict(float)
    for ch_idx, results in enumerate(result_lists):
        w = float(weights[ch_idx]) if weights is not None else 1.0
        for rank, (item_id, _) in enumerate(results):
            scores[item_id] += w * 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def priority_fusion(*result_lists: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Concatenate ranked lists, preserving the first list's order exactly.

    RRF treats every channel's rank-1 hit as interchangeable with every other
    channel's rank-1 hit. That is only sound when the channels are comparably
    good. Measured on the real labelled query set they are not: BM25 alone
    scores MRR 0.2429 and P@1 0.1190, while the best six-channel RRF
    configuration scores 0.2272 and 0.0635. Fusion was reordering a good list
    using worse ones.

    Here the leading channel keeps its ranking and the rest only contribute
    candidates it never returned, appended in their own order. Recall still
    rises; precision at the head is no longer traded away for it.

    The returned score is a positional proxy so downstream consumers (the
    confidence gate, feature extraction) keep a monotonically decreasing value.
    """
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for results in result_lists:
        for item_id, _ in results:
            if item_id in seen:
                continue
            seen.add(item_id)
            out.append((item_id, 1.0 / len(out) if out else 1.0))
    return out


# ── Main search pipeline ─────────────────────────────────────

async def hybrid_search(
    store: Store,
    query: str,
    embedding_provider: EmbeddingProvider | None,
    *,
    limit: int = 10,
    current_only: bool = True,
    as_of: str | None = None,
    rerank: bool = False,
    profile: UserProfile | None = None,
    session: SessionState | None = None,
    ann_index: ANNIndex | None = None,
    min_score: float = 0.0,
    diverse: bool = False,
    preranker: PreRanker | None = None,
    online_ranker: OnlineRanker | None = None,
    blender: ThompsonBlender | None = None,
    activation_weighting: bool = True,
    query_independent_channels: bool = False,
    fusion: str = "priority",
) -> list[SearchResult]:
    """Orchestrate 6-channel search (BM25 + embedding + graph + user interest + time + session), fuse with RRF, optionally rerank."""
    # The candidate pool is sized by what retrieval needs, not by whether a
    # reranker happens to be installed. Tying the two together meant that with
    # reranking off the pipeline kept only RRF's top `limit`: measured on the
    # real labelled query set, 38/126 answers found versus 80/126 for the same
    # fusion given the full pool.
    reranker_available = rerank and _get_reranker() is not None
    fetch_limit = limit * 5

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

    # Channels 4-6 do not read the query at all: they return recent facts, the
    # user's habitual entities, and the current session's entities. Under RRF a
    # fact ranked #1 by recency contributes exactly as much as one ranked #1 by
    # BM25, and a fact present in all three outscores the best keyword match
    # threefold. Measured at the fusion stage on the real labelled query set,
    # switching them off took answer-present from 38/126 to 80/126 and MRR from
    # 0.0917 to 0.1692. They are off by default and opt-in for callers that
    # genuinely want ambient recall rather than an answer to the query.
    # 4. User interest channel
    interest_results: list[tuple[int, float]] = []
    if query_independent_channels and profile and profile.entity_affinity:
        interest_results = await recall_user_interest(store, profile, limit=fetch_limit)

    # 5. Time window channel
    time_results = (
        await recall_time_window(store, days=7, limit=fetch_limit)
        if query_independent_channels else []
    )

    # 6. Session context channel
    session_results: list[tuple[int, float]] = []
    if query_independent_channels and session and session.entity_activation:
        session_results = await recall_session_context(store, session, limit=fetch_limit)

    # RRF fusion — optionally with Thompson-sampled channel weights
    all_channels = [bm25_results, emb_results, graph_results,
                    interest_results, time_results, session_results]
    # Track which original channel indices are non-empty (for weighted RRF)
    active_indices = [i for i, r in enumerate(all_channels) if r]
    sources_to_fuse = [all_channels[i] for i in active_indices]
    if not sources_to_fuse:
        return []

    if fusion == "rrf":
        channel_weights: np.ndarray | None = None
        if blender is not None and blender.n_updates > 0:
            all_weights = blender.sample_weights()
            channel_weights = np.array([all_weights[i] for i in active_indices])
        fused = reciprocal_rank_fusion(*sources_to_fuse, weights=channel_weights)
    else:
        fused = priority_fusion(*sources_to_fuse)

    # Build per-fact channel origin map: fact_id -> set of original channel indices
    _fact_channels: dict[int, set[int]] = defaultdict(set)
    for ch_idx, results in zip(active_indices, sources_to_fuse):
        for fid, _ in results:
            _fact_channels[fid].add(ch_idx)

    # 5. Load facts, filter, score
    candidate_limit = fetch_limit
    results: list[SearchResult] = []
    # Collect per-channel scores for feature extraction
    bm25_scores: dict[int, float] = {fid: s for fid, s in bm25_results}
    emb_scores: dict[int, float] = {fid: s for fid, s in emb_results}
    # Graph distance: lower rank = closer.  Convert rank to distance proxy.
    graph_dists: dict[int, float] = {}
    for rank, (fid, _) in enumerate(graph_results):
        graph_dists[fid] = float(rank)

    as_of_dt = datetime.fromisoformat(as_of) if as_of else None

    # Resolve the fused candidates in one round-trip rather than one await each.
    facts_by_id = await store.get_facts(fid for fid, _ in fused)
    kept: list[tuple[Fact, float]] = []
    for fact_id, rrf_score in fused:
        fact = facts_by_id.get(fact_id)
        if not fact or not filter_temporal([fact], current_only, as_of):
            continue
        kept.append((fact, rrf_score))
        if len(kept) >= candidate_limit:
            break

    entities_by_id = await store.get_entities(f.entity_id for f, _ in kept if f.entity_id)

    # Recency/frequency is already channel 5 of the fusion above. Multiplying
    # the fused score by ACT-R activation as well double-counts it, and the
    # activation term spans roughly two orders of magnitude: a fact with
    # access_count=0 that is 150 days old scores ~0.04, so any older memory is
    # driven below every recent one regardless of how well it matches the
    # query. Measured on the real labelled query set, this alone dropped
    # answer-present from 98% (BM25 top-50) to 28% (fused top-15).
    for fact, rrf_score in kept:
        if activation_weighting:
            importance_boost = 1.0 + fact.importance * 2  # importance 0.9 → 2.8x boost
            final_score = rrf_score * temporal_score(fact, as_of=as_of_dt) * importance_boost
        else:
            final_score = rrf_score

        results.append(SearchResult(
            fact=fact, score=final_score, source="fused",
            entity=entities_by_id.get(fact.entity_id) if fact.entity_id else None,
            channels=_fact_channels.get(fact.id, set()),
        ))

    # 5b. PreRanker: re-sort candidates using LightGBM (when model loaded)
    if preranker is not None and preranker.is_loaded and len(results) > 1:
        from maestro_memory.ranking.features import extract_features
        feat_matrix = np.array([
            extract_features(
                query=query,
                fact_content=r.fact.content,
                fact_importance=r.fact.importance,
                fact_access_count=r.fact.access_count,
                fact_created_at=r.fact.created_at,
                fact_last_accessed=r.fact.last_accessed,
                fact_entity_id=r.fact.entity_id,
                bm25_score=bm25_scores.get(r.fact.id, 0.0),
                embed_score=emb_scores.get(r.fact.id, 0.0),
                graph_distance=graph_dists.get(r.fact.id, -1.0),
                entity_affinity=(
                    profile.get_affinity(r.fact.entity_id) if profile and r.fact.entity_id else 0.0
                ),
                session_boost=(
                    session.entity_activation.get(r.fact.entity_id, 0.0)
                    if session and r.fact.entity_id else 0.0
                ),
            )
            for r in results
        ])
        candidate_ids = [r.fact.id for r in results]
        ranked_pairs = preranker.rank(feat_matrix, candidate_ids, limit=len(results))
        id_to_rank = {cid: idx for idx, (cid, _) in enumerate(ranked_pairs)}
        results.sort(key=lambda r: id_to_rank.get(r.fact.id, len(results)))
        # Update scores from preranker
        id_to_score = {cid: sc for cid, sc in ranked_pairs}
        for r in results:
            if r.fact.id in id_to_score:
                r.score = id_to_score[r.fact.id]

    # 5c. OnlineRanker: boost scores using streaming P(used) prediction
    if online_ranker is not None and online_ranker.n_updates > 0 and len(results) > 0:
        from maestro_memory.ranking.features import extract_features, features_to_dict
        for r in results:
            feats = extract_features(
                query=query,
                fact_content=r.fact.content,
                fact_importance=r.fact.importance,
                fact_access_count=r.fact.access_count,
                fact_created_at=r.fact.created_at,
                fact_last_accessed=r.fact.last_accessed,
                fact_entity_id=r.fact.entity_id,
                bm25_score=bm25_scores.get(r.fact.id, 0.0),
                embed_score=emb_scores.get(r.fact.id, 0.0),
                graph_distance=graph_dists.get(r.fact.id, -1.0),
                entity_affinity=(
                    profile.get_affinity(r.fact.entity_id) if profile and r.fact.entity_id else 0.0
                ),
                session_boost=(
                    session.entity_activation.get(r.fact.entity_id, 0.0)
                    if session and r.fact.entity_id else 0.0
                ),
            )
            online_pred = online_ranker.predict(features_to_dict(feats))
            r.score *= (0.5 + online_pred)  # range 0.5x–1.5x
        results.sort(key=lambda r: -r.score)

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
