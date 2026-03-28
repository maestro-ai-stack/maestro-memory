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
class SearchMeta:
    """Search quality metadata — embedded in output to guide agent behavior."""
    confidence: str = "high"  # high | medium | low | none
    best_score: float = 0.0
    hint: str = ""
    suggestion: str | None = None  # suggested follow-up search

    @staticmethod
    def from_results(query: str, results: list[SearchResult], threshold: float = 0.001) -> SearchMeta:
        if not results:
            return SearchMeta(
                confidence="none", best_score=0.0,
                hint="No relevant data found. This topic may not be in memory.",
                suggestion=None,
            )

        best = results[0].score
        entities = {r.fact.entity_id for r in results if r.fact.entity_id}
        entity_names = {r.entity.name for r in results if r.entity}
        query_lower = query.lower()

        # Confidence based on score relative to threshold
        if best < threshold * 2:
            confidence = "low"
            hint = "Results have very low relevance. This topic may not be in memory."
        elif best < threshold * 10:
            confidence = "medium"
            hint = "Results have moderate relevance."
        else:
            confidence = "high"
            hint = ""

        suggestion = None

        # Aggregation detection
        agg_words = {"all", "list", "every", "how many", "what are the", "conditions", "complete", "total"}
        if any(w in query_lower for w in agg_words):
            if len(entities) <= 2 and len(results) >= 3:
                hint = "Results cluster around few topics. Run additional searches to gather scattered facts."
                if entity_names:
                    suggestion = f"Try narrower searches for specific aspects of: {query}"
            elif len(results) < 5:
                hint = "Few results found. There may be more relevant facts under different search terms."

        return SearchMeta(
            confidence=confidence,
            best_score=best,
            hint=hint.strip(),
            suggestion=suggestion,
        )


@dataclass
class AddResult:
    episode_id: int
    facts_added: int = 0
    facts_updated: int = 0
    facts_invalidated: int = 0
    entities_created: int = 0
