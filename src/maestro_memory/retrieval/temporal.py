from __future__ import annotations

import math
from datetime import datetime

from maestro_memory.core.models import Fact


def temporal_score(fact: Fact, as_of: datetime | None = None) -> float:
    """Ebbinghaus-inspired temporal decay score."""
    base = fact.importance
    now = as_of or datetime.utcnow()

    # Parse created_at
    try:
        created = datetime.fromisoformat(fact.created_at)
    except (ValueError, TypeError):
        created = now

    # Recency decay
    days_old = max((now - created).total_seconds() / 86400, 0)
    recency = math.exp(-0.01 * days_old)

    # Access reinforcement
    access_boost = 1 + fact.access_count * 0.1

    return base * recency * access_boost


def filter_temporal(
    facts: list[Fact],
    current_only: bool,
    as_of: str | None = None,
) -> list[Fact]:
    """Filter facts by temporal validity."""
    if not current_only and not as_of:
        return facts

    now_str = as_of or datetime.utcnow().isoformat()
    result = []
    for f in facts:
        # If current_only, exclude expired facts
        if current_only and f.valid_until is not None:
            if f.valid_until < now_str:
                continue
        # If as_of, only include facts valid at that time
        if as_of:
            if f.valid_from > as_of:
                continue
            if f.valid_until and f.valid_until < as_of:
                continue
        result.append(f)
    return result
