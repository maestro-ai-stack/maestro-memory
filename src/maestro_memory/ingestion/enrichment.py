"""Context string enrichment for better embeddings (Phase 0).

Enriches fact content with category, related concepts, causal links,
and importance signal before embedding. Original content is preserved —
only the embedding vector is computed from the enriched version.
"""

from __future__ import annotations

# Map fact_type codes to human-readable labels
_FACT_TYPE_LABELS: dict[str, str] = {
    "observation": "general observation",
    "preference": "user preference",
    "feedback": "user feedback",
    "decision": "strategic decision",
    "rule": "business rule",
    "metric": "metric definition",
    "guidance": "system guidance",
}

# Map predicate verbs to grouping labels for relation formatting
_PREDICATE_GROUPS: dict[str, str] = {
    "causes": "Causes",
    "affects": "Affects",
    "derived_from": "Derived from",
    "depends_on": "Depends on",
    "collaborates_with": "Collaborates with",
    "works_at": "Works at",
    "uses": "Uses",
    "needs": "Needs",
    "produces": "Produces",
    "part_of": "Part of",
}


def _importance_label(importance: float) -> str:
    if importance > 0.8:
        return "CRITICAL"
    if importance > 0.6:
        return "HIGH"
    if importance > 0.3:
        return "NORMAL"
    return "LOW"


def enrich_template(
    content: str,
    *,
    fact_type: str = "observation",
    entity_name: str | None = None,
    entity_type: str = "concept",
    importance: float = 0.5,
    related_entities: list[str] | None = None,
    relations: list[tuple[str, str]] | None = None,
) -> str:
    """Fast template-based enrichment (no LLM, no async).

    Appends structured metadata to *content* so the embedding model
    captures category, entity, relations, and importance signals.
    The original content is the first line — everything after is context.
    """
    parts: list[str] = [content]

    # Category
    label = _FACT_TYPE_LABELS.get(fact_type, fact_type)
    parts.append(f"Category: {label}.")

    # Entity
    if entity_name:
        parts.append(f"Entity: {entity_name} ({entity_type}).")

    # Related entities
    if related_entities:
        parts.append(f"Related: {', '.join(related_entities)}.")

    # Relations — group by predicate for readability
    if relations:
        grouped: dict[str, list[str]] = {}
        for predicate, target in relations:
            group_label = _PREDICATE_GROUPS.get(predicate, predicate.replace("_", " ").capitalize())
            grouped.setdefault(group_label, []).append(target)
        for group_label, targets in grouped.items():
            parts.append(f"{group_label}: {', '.join(targets)}.")

    # Importance signal
    parts.append(f"Importance: {_importance_label(importance)}.")

    return "\n".join(parts)


async def enrich_for_embedding(
    content: str,
    *,
    fact_type: str = "observation",
    entity_name: str | None = None,
    entity_type: str = "concept",
    importance: float = 0.5,
    related_entities: list[str] | None = None,
    relations: list[tuple[str, str]] | None = None,
) -> str:
    """Generate enriched context string for embedding.

    Async wrapper around :func:`enrich_template` — kept async so a future
    Phase 3 can drop in LLM-based enrichment without changing callers.

    The enriched string includes category, related concepts,
    causal/relational links, and importance signal.
    Original content is NOT modified — only the embedding input changes.
    """
    return enrich_template(
        content,
        fact_type=fact_type,
        entity_name=entity_name,
        entity_type=entity_type,
        importance=importance,
        related_entities=related_entities,
        relations=relations,
    )
