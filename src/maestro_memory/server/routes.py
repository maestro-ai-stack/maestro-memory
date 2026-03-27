"""HTTP routes for the mmem daemon."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from maestro_memory.server.lifecycle import get_memory

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    current_only: bool = True
    as_of: str | None = None
    rerank: bool = True


class SearchResultItem(BaseModel):
    fact_id: int
    content: str
    score: float
    entity_name: str | None = None
    fact_type: str = ""
    importance: float = 0.5


class AddRequest(BaseModel):
    content: str
    source_type: str = "manual"
    source_ref: str | None = None
    fact_type: str = "observation"
    importance: float = 0.5
    entity_name: str | None = None
    entity_type: str = "concept"


class AddResponse(BaseModel):
    episode_id: int
    facts_added: int = 0
    facts_updated: int = 0
    facts_invalidated: int = 0
    entities_created: int = 0


@router.get("/health")
async def health():
    mem = get_memory()
    stats = await mem.status()
    return {"status": "ok", **stats}


@router.post("/search")
async def search(req: SearchRequest) -> list[SearchResultItem]:
    mem = get_memory()
    results = await mem.search(
        req.query,
        limit=req.limit,
        current_only=req.current_only,
        as_of=req.as_of,
        rerank=req.rerank,
    )
    return [
        SearchResultItem(
            fact_id=r.fact.id,
            content=r.fact.content,
            score=r.score,
            entity_name=r.entity.name if r.entity else None,
            fact_type=r.fact.fact_type,
            importance=r.fact.importance,
        )
        for r in results
    ]


@router.post("/add")
async def add(req: AddRequest) -> AddResponse:
    mem = get_memory()
    result = await mem.add(
        req.content,
        source_type=req.source_type,
        source_ref=req.source_ref,
        fact_type=req.fact_type,
        importance=req.importance,
        entity_name=req.entity_name,
        entity_type=req.entity_type,
    )
    return AddResponse(
        episode_id=result.episode_id,
        facts_added=result.facts_added,
        facts_updated=result.facts_updated,
        facts_invalidated=result.facts_invalidated,
        entities_created=result.entities_created,
    )


@router.get("/status")
async def status():
    mem = get_memory()
    return await mem.status()
