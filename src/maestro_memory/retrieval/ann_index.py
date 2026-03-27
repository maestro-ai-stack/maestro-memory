from __future__ import annotations

import numpy as np


class ANNIndex:
    """HNSW approximate nearest neighbor index, lazy-loaded."""

    def __init__(self, dim: int = 384, max_elements: int = 100_000):
        self._dim = dim
        self._max = max_elements
        self._index = None
        self._id_map: dict[int, int] = {}  # fact_id -> internal_id
        self._reverse: dict[int, int] = {}  # internal_id -> fact_id
        self._count = 0

    def _ensure_index(self) -> bool:
        if self._index is not None:
            return True
        try:
            import hnswlib

            self._index = hnswlib.Index(space="cosine", dim=self._dim)
            self._index.init_index(max_elements=self._max, ef_construction=200, M=16)
            self._index.set_ef(50)
            return True
        except ImportError:
            return False

    def add(self, fact_id: int, embedding: np.ndarray) -> None:
        if not self._ensure_index():
            return
        if fact_id in self._id_map:
            return
        internal_id = self._count
        self._index.add_items(embedding.reshape(1, -1).astype(np.float32), np.array([internal_id]))
        self._id_map[fact_id] = internal_id
        self._reverse[internal_id] = fact_id
        self._count += 1

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if self._index is None or self._count == 0:
            return []
        k = min(k, self._count)
        labels, distances = self._index.knn_query(query.reshape(1, -1).astype(np.float32), k=k)
        results = []
        for internal_id, dist in zip(labels[0], distances[0]):
            fact_id = self._reverse.get(int(internal_id))
            if fact_id is not None:
                similarity = 1.0 - dist  # cosine distance -> similarity
                results.append((fact_id, float(similarity)))
        return results

    @property
    def size(self) -> int:
        return self._count
