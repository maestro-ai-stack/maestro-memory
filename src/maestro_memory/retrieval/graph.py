from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro_memory.core.store import Store


async def graph_neighbors(
    store: Store,
    entity_ids: list[int],
    hops: int = 1,
    current_only: bool = True,
) -> list[tuple[int, float]]:
    """Find facts connected to entities within N hops.

    Returns list of (fact_id, score) where score decays with hop distance.
    """
    if not entity_ids:
        return []

    # Iterative BFS to collect entities within N hops
    visited: set[int] = set(entity_ids)
    frontier: set[int] = set(entity_ids)
    entity_scores: dict[int, float] = {eid: 1.0 for eid in entity_ids}

    for hop in range(1, hops + 1):
        next_frontier: set[int] = set()
        for eid in frontier:
            relations = await store.get_relations_for_entity(eid, current_only=current_only)
            for rel in relations:
                neighbor = rel.object_id if rel.subject_id == eid else rel.subject_id
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    # Decay score by hop distance
                    entity_scores[neighbor] = 1.0 / (hop + 1)
        frontier = next_frontier

    # Collect facts linked to any discovered entity
    fact_scores: dict[int, float] = {}
    for eid, score in entity_scores.items():
        facts = await _facts_for_entity(store, eid, current_only)
        for fid in facts:
            if fid not in fact_scores or fact_scores[fid] < score:
                fact_scores[fid] = score

    return sorted(fact_scores.items(), key=lambda x: -x[1])


async def _facts_for_entity(store: Store, entity_id: int, current_only: bool) -> list[int]:
    """Return fact IDs linked to an entity."""
    sql = "SELECT id FROM facts WHERE entity_id = ?"
    params: list = [entity_id]
    if current_only:
        sql += " AND valid_until IS NULL"
    cur = await store.db.execute(sql, params)
    rows = await cur.fetchall()
    return [row[0] for row in rows]
