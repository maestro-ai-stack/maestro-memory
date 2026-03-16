from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    id: int
    name: str
    entity_type: str = "concept"
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Relation:
    id: int
    subject_id: int
    predicate: str
    object_id: int
    confidence: float = 1.0
    valid_from: str = ""
    valid_until: str | None = None
    episode_id: int | None = None


@dataclass
class Fact:
    id: int
    content: str
    fact_type: str = "observation"
    importance: float = 0.5
    entity_id: int | None = None
    episode_id: int | None = None
    valid_from: str = ""
    valid_until: str | None = None
    access_count: int = 0
    last_accessed: str | None = None
    created_at: str = ""


@dataclass
class Episode:
    id: int
    content: str
    source_type: str = "manual"
    source_ref: str | None = None
    created_at: str = ""


@dataclass
class SearchResult:
    fact: Fact
    score: float
    source: str = ""  # "bm25" | "embedding" | "graph" | "fused"
    entity: Entity | None = None


@dataclass
class AddResult:
    episode_id: int
    facts_added: int = 0
    facts_updated: int = 0
    facts_invalidated: int = 0
    entities_created: int = 0
