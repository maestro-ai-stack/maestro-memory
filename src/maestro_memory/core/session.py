"""Session state for implicit context-aware search.

Tracks recent queries, accessed facts, and entity affinity within a session.
Enables context-boosted retrieval without explicit parameters from the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SessionState:
    """Ephemeral per-session state for context-aware retrieval.

    Updated automatically on every search() and add() call.
    Not persisted — cleared when Memory is closed.
    """

    max_queries: int = 20
    max_facts: int = 50
    embedding_dim: int = 384  # updated on first embedding seen
    decay: float = 0.9  # EMA decay for session embedding

    # Recent queries (FIFO)
    recent_queries: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    # Recent fact IDs returned by search (FIFO)
    recent_fact_ids: deque[int] = field(default_factory=lambda: deque(maxlen=50))

    # Entity activation: entity_id → cumulative activation (decayed)
    entity_activation: dict[int, float] = field(default_factory=dict)

    # Running average embedding of queries (session topic vector)
    _session_embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def session_embedding(self) -> np.ndarray | None:
        return self._session_embedding

    def record_query(self, query: str, query_embedding: np.ndarray | None = None) -> None:
        """Record a search query and update session state."""
        self.recent_queries.append(query)

        if query_embedding is not None:
            if self._session_embedding is None:
                self._session_embedding = query_embedding.copy()
                self.embedding_dim = len(query_embedding)
            else:
                self._session_embedding = (
                    self.decay * self._session_embedding
                    + (1 - self.decay) * query_embedding
                )

    def record_results(self, fact_ids: list[int], entity_ids: list[int | None]) -> None:
        """Record search results to track accessed facts and entity affinity."""
        for fid in fact_ids:
            self.recent_fact_ids.append(fid)

        for eid in entity_ids:
            if eid is not None:
                current = self.entity_activation.get(eid, 0.0)
                self.entity_activation[eid] = current * 0.95 + 1.0

    def record_add(self, fact_ids: list[int], entity_ids: list[int | None]) -> None:
        """Record an add operation (stronger signal than search results)."""
        for fid in fact_ids:
            self.recent_fact_ids.append(fid)

        for eid in entity_ids:
            if eid is not None:
                current = self.entity_activation.get(eid, 0.0)
                self.entity_activation[eid] = current * 0.95 + 2.0

    def get_entity_boost(self, entity_id: int | None) -> float:
        """Get activation boost for an entity (0.0 if not activated)."""
        if entity_id is None:
            return 0.0
        return self.entity_activation.get(entity_id, 0.0)

    def expand_query(self, query: str) -> str:
        """Expand query with recent session context (lightweight, no LLM).

        Appends recent query keywords to improve recall for follow-up questions.
        E.g., query="hotel" with recent_queries=["Tokyo trip planning"]
              → "hotel Tokyo trip planning"
        """
        if not self.recent_queries:
            return query

        # Take last 3 queries, extract unique words not already in query
        query_words = set(query.lower().split())
        context_words: list[str] = []
        for prev_query in list(self.recent_queries)[-3:]:
            for word in prev_query.split():
                if word.lower() not in query_words and len(word) > 2:
                    context_words.append(word)
                    query_words.add(word.lower())

        if not context_words:
            return query

        # Append top context words (max 10)
        expansion = " ".join(context_words[:10])
        return f"{query} {expansion}"

    def similarity_to_session(self, fact_embedding: np.ndarray) -> float:
        """Cosine similarity between a fact embedding and the session topic vector."""
        if self._session_embedding is None:
            return 0.0
        dot = np.dot(self._session_embedding, fact_embedding)
        norm_a = np.linalg.norm(self._session_embedding)
        norm_b = np.linalg.norm(fact_embedding)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def reset(self) -> None:
        """Clear all session state."""
        self.recent_queries.clear()
        self.recent_fact_ids.clear()
        self.entity_activation.clear()
        self._session_embedding = None
