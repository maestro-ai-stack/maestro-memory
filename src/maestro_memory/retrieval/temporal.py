from __future__ import annotations

import math
from datetime import datetime

from maestro_memory.core.models import Fact


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ── ACT-R activation model ────────────────────────────────

_SPREADING_WEIGHT = 0.3


def temporal_score(
    fact: Fact,
    as_of: datetime | None = None,
    similarity: float = 0.0,
) -> float:
    """ACT-R activation: A = B(recency, frequency) + w * S(similarity)."""
    now = as_of or datetime.now(tz=None)

    # Parse creation time
    try:
        created = datetime.fromisoformat(fact.created_at)
    except (ValueError, TypeError):
        created = now

    days_old = max((now - created).total_seconds() / 86400, 0)

    # Base-level activation: frequency gain - time decay
    base_level = math.log(fact.access_count + 1) - 0.5 * math.log(days_old + 1)

    # Total activation = base-level + spreading activation
    activation = base_level + _SPREADING_WEIGHT * similarity

    return fact.importance * _sigmoid(activation)


def filter_temporal(
    facts: list[Fact],
    current_only: bool,
    as_of: str | None = None,
) -> list[Fact]:
    """Filter facts by temporal validity."""
    if not current_only and not as_of:
        return facts

    now_str = as_of or datetime.now(tz=None).isoformat()
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
