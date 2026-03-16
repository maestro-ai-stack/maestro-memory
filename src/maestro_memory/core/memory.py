from __future__ import annotations

from pathlib import Path

from maestro_memory.core.config import get_db_path
from maestro_memory.core.models import AddResult, SearchResult
from maestro_memory.core.store import Store
from maestro_memory.ingestion.extractor import llm_extract
from maestro_memory.ingestion.fallback import fallback_extract
from maestro_memory.retrieval.embedding import get_embedding_provider
from maestro_memory.retrieval.fusion import hybrid_search


class Memory:
    """Main facade (entrypoint) for maestro-memory."""

    def __init__(self, path: str | Path | None = None, project: str | None = None) -> None:
        if path is not None:
            self._db_path = Path(path)
        else:
            self._db_path = get_db_path(project)
        self.store = Store(self._db_path)
        self._embedding_provider = None

    async def init(self) -> None:
        """Open store and create tables."""
        await self.store.init()
        self._embedding_provider = get_embedding_provider()

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

                emb_bytes = None
                if self._embedding_provider:
                    emb = await self._embedding_provider.embed(fact_content)
                    if emb is not None:
                        emb_bytes = emb.tobytes()

                await self.store.add_fact(
                    content=fact_content,
                    fact_type=op.get("type", fact_type),
                    importance=op.get("importance", importance),
                    embedding=emb_bytes,
                    entity_id=entity_id,
                    episode_id=episode_id,
                )
                result.facts_added += 1

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
    ) -> list[SearchResult]:
        """Hybrid search pipeline."""
        return await hybrid_search(
            self.store, query, self._embedding_provider,
            limit=limit, current_only=current_only, as_of=as_of,
        )

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
        """Close store."""
        await self.store.close()
