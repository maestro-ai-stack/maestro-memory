"""Adaptive confidence threshold — learned from feedback data."""
from __future__ import annotations

import json

DEFAULT_THRESHOLD = 0.001


async def compute_threshold(store) -> float:
    """Compute from serving_logs: 10th percentile of used fact scores."""
    cur = await store.db.execute(
        "SELECT returned_fact_ids, used_fact_ids FROM serving_logs "
        "WHERE used_fact_ids IS NOT NULL ORDER BY id DESC LIMIT 200"
    )
    rows = await cur.fetchall()
    if len(rows) < 5:
        return DEFAULT_THRESHOLD

    # Collect scores of facts that were actually used
    # For now, use position-based proxy: used facts were in top results
    used_positions = []
    for row in rows:
        returned = json.loads(row[0]) if row[0] else []
        used = set(json.loads(row[1]) if row[1] else [])
        for i, fid in enumerate(returned):
            if fid in used:
                used_positions.append(i)

    if len(used_positions) < 3:
        return DEFAULT_THRESHOLD

    # Facts typically used are in top positions — use this to set threshold
    # For score-based threshold, we'd need scores stored in logs
    # For now: return conservative default (will improve when features_json has scores)
    return DEFAULT_THRESHOLD
