from __future__ import annotations

from pathlib import Path

from maestro_memory.core.config import get_db_path
from maestro_memory.core.models import AddResult, SearchResult
from maestro_memory.core.profile import UserProfile
from maestro_memory.core.session import SessionState
from maestro_memory.core.store import Store
from maestro_memory.ingestion.extractor import llm_extract
from maestro_memory.ingestion.fallback import fallback_extract
from maestro_memory.logging.serving_log import ServingLogger
from maestro_memory.ranking.blender import ThompsonBlender
from maestro_memory.ranking.online import OnlineRanker
from maestro_memory.ranking.prerank import PreRanker
import numpy as np

from maestro_memory.retrieval.ann_index import ANNIndex
from maestro_memory.retrieval.embedding import get_embedding_provider
from maestro_memory.retrieval.fusion import hybrid_search, set_ann_index


class Memory:
    """Main facade (entrypoint) for maestro-memory."""

    def __init__(self, path: str | Path | None = None, project: str | None = None) -> None:
        if path is not None:
            self._db_path = Path(path)
        else:
            self._db_path = get_db_path(project)
        self.store = Store(self._db_path)
        self._embedding_provider = None
        self.session = SessionState()
        self.profile = UserProfile()
        self._preranker = PreRanker()
        self._online_ranker = OnlineRanker()
        self._blender = ThompsonBlender()
        self._serving_logger: ServingLogger | None = None

    async def init(self) -> None:
        """Open store and create tables, build ANN index from existing embeddings."""
        await self.store.init()
        self._serving_logger = ServingLogger(self.store)
        self.profile = await self.store.load_profile()
        self._embedding_provider = get_embedding_provider()

        # Build ANN index from existing embeddings
        cur = await self.store.db.execute("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")
        rows = await cur.fetchall()
        if rows:
            dim = len(np.frombuffer(rows[0][1], dtype=np.float32))
            ann = ANNIndex(dim=dim)
            for row in rows:
                emb = np.frombuffer(row[1], dtype=np.float32)
                ann.add(row[0], emb)
            set_ann_index(ann)
            self._ann_index = ann
        else:
            self._ann_index = None

    async def add(
        self,
        content: str,
        *,
        source_type: str = "manual",
        source_ref: str | None = None,
        fact_type: str = "observation",
        importance: float = 0.5,
        entity_name: str | None = None,
        entity_type: str = "concept",
    ) -> AddResult:
        """Ingest content into memory.

        Agent can specify entity_name directly, skipping LLM extraction.
        """
        episode_id = await self.store.add_episode(content, source_type, source_ref)

        # Agent provides entity directly -> skip LLM extraction
        if entity_name:
            operations = [{"op": "ADD", "fact": content, "entity": entity_name,
                          "entity_type": entity_type, "type": fact_type, "importance": importance}]
        else:
            # Try LLM extraction first, then fallback
            existing = await self.store.list_entities()
            operations = await llm_extract(content, existing)
            if not operations:
                operations = await fallback_extract(content)

        result = AddResult(episode_id=episode_id)

        for op in operations:
            action = op.get("op", "ADD")
            if action == "ADD":
                fact_content = op.get("fact", content)
                entity_name = op.get("entity")
                entity_id = None
                if entity_name:
                    entity, created = await self.store.get_or_create_entity(entity_name, op.get("entity_type", "concept"))
                    entity_id = entity.id
                    if created:
                        result.entities_created += 1

                emb = None
                emb_bytes = None
                if self._embedding_provider:
                    emb = await self._embedding_provider.embed(fact_content)
                    if emb is not None:
                        emb_bytes = emb.tobytes()

                fact_id = await self.store.add_fact(
                    content=fact_content,
                    fact_type=op.get("type", fact_type),
                    importance=op.get("importance", importance),
                    embedding=emb_bytes,
                    entity_id=entity_id,
                    episode_id=episode_id,
                )
                result.facts_added += 1

                # Add to ANN index
                if emb is not None:
                    if self._ann_index is None:
                        self._ann_index = ANNIndex(dim=len(emb))
                        set_ann_index(self._ann_index)
                    self._ann_index.add(fact_id, emb)

            elif action == "UPDATE":
                fid = op.get("fact_id")
                if fid and op.get("new_content"):
                    await self.store.update_fact(fid, op["new_content"])
                    result.facts_updated += 1

            elif action == "INVALIDATE":
                fid = op.get("fact_id")
                if fid:
                    await self.store.invalidate_fact(fid)
                    result.facts_invalidated += 1

        return result

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        current_only: bool = True,
        as_of: str | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search pipeline with optional cross-encoder reranking.

        Automatically updates session state for context-aware follow-up queries.
        """
        # Get query embedding for session tracking
        query_emb = None
        if self._embedding_provider:
            query_emb = await self._embedding_provider.embed(query)

        async with self._serving_logger.log_search(query) as log_entry:
            results = await hybrid_search(
                self.store, query, self._embedding_provider,
                limit=limit, current_only=current_only, as_of=as_of,
                rerank=rerank,
                profile=self.profile, session=self.session,
                ann_index=self._ann_index,
            )
            log_entry["candidate_ids"] = [r.fact.id for r in results]
            log_entry["returned_ids"] = [r.fact.id for r in results[:limit]]

            # Compute 12-dim features for logging (does not change ranking)
            import json
            from maestro_memory.ranking.features import extract_features, FEATURE_NAMES
            features_list = []
            for r in results:
                feats = extract_features(
                    query=query,
                    fact_content=r.fact.content,
                    fact_importance=r.fact.importance,
                    fact_access_count=r.fact.access_count,
                    fact_created_at=r.fact.created_at,
                    fact_last_accessed=r.fact.last_accessed,
                    fact_entity_id=r.fact.entity_id,
                    bm25_score=r.score,  # approximate — RRF fused score
                    embed_score=0.0,
                    graph_distance=0.0,
                    entity_affinity=self.profile.get_affinity(r.fact.entity_id) if r.fact.entity_id else 0.0,
                    session_boost=self.session.entity_activation.get(r.fact.entity_id, 0.0) if r.fact.entity_id else 0.0,
                )
                features_list.append(
                    {name: float(feats[i]) for i, name in enumerate(FEATURE_NAMES)}
                )
            log_entry["features_json"] = json.dumps(features_list)

        # Auto-update session state
        self.session.record_query(query, query_emb)
        self.session.record_results(
            [r.fact.id for r in results],
            [r.fact.entity_id for r in results],
        )

        # Auto-update user profile
        self.profile.total_searches += 1
        if query_emb is not None:
            self.profile.update_topic(query_emb)
        for r in results:
            if r.fact.entity_id is not None:
                self.profile.update_entity(r.fact.entity_id)

        return results

    async def graph(self, entity_name: str, *, hops: int = 1) -> dict:  # noqa: ARG002
        """Graph traversal from an entity (multi-hop planned for future)."""
        entity = await self.store.get_entity_by_name(entity_name)
        if not entity:
            return {"entity": None, "relations": [], "neighbors": []}
        relations = await self.store.get_relations_for_entity(entity.id)
        neighbor_ids = set()
        for r in relations:
            neighbor_ids.add(r.subject_id if r.subject_id != entity.id else r.object_id)

        neighbors = []
        for nid in neighbor_ids:
            n = await self.store.get_entity(nid)
            if n:
                neighbors.append(n)

        return {
            "entity": entity,
            "relations": relations,
            "neighbors": neighbors,
        }

    async def status(self) -> dict:
        """Return DB stats."""
        return await self.store.get_stats()

    async def close(self) -> None:
        """Persist profile and close store."""
        await self.store.save_profile(self.profile)
        await self.store.close()
