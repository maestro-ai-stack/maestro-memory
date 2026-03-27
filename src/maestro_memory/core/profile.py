from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UserProfile:
    """3-layer user profile: long-term (entity affinity, topic embedding),
    mid-term (Ebbinghaus decay), short-term (SessionState - already exists)."""

    entity_affinity: dict[int, float] = field(default_factory=dict)
    topic_embedding: np.ndarray | None = field(default=None, repr=False)
    total_searches: int = 0
    total_adds: int = 0

    def update_entity(self, entity_id: int, boost: float = 1.0) -> None:
        old = self.entity_affinity.get(entity_id, 0.0)
        self.entity_affinity[entity_id] = old * 0.95 + boost

    def get_affinity(self, entity_id: int) -> float:
        return self.entity_affinity.get(entity_id, 0.0)

    def decay_all(self, factor: float = 0.85) -> None:
        for eid in list(self.entity_affinity):
            self.entity_affinity[eid] *= factor
            if self.entity_affinity[eid] < 0.01:
                del self.entity_affinity[eid]

    def update_topic(self, query_embedding: np.ndarray, alpha: float = 0.05) -> None:
        if self.topic_embedding is None:
            self.topic_embedding = query_embedding.copy()
        else:
            self.topic_embedding = (1 - alpha) * self.topic_embedding + alpha * query_embedding

    def top_entities(self, n: int = 10) -> list[tuple[int, float]]:
        return sorted(self.entity_affinity.items(), key=lambda x: -x[1])[:n]
