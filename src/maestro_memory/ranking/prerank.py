"""LightGBM pre-ranker using 12-dim features."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class PreRanker:
    """LightGBM pre-ranker. Falls back to BM25 score if no model loaded."""

    def __init__(self, model_path: Path | None = None):
        self._model = None
        self._model_path = model_path

    def load(self) -> bool:
        if self._model_path and self._model_path.exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(self._model_path))
                return True
            except ImportError:
                return False
        return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score candidates. features shape: (n_candidates, 12)."""
        if self._model is None:
            return features[:, 0] if features.ndim == 2 else features
        return self._model.predict(features)

    def rank(self, features: np.ndarray, candidate_ids: list[int], limit: int) -> list[tuple[int, float]]:
        """Return top-limit (candidate_id, score) pairs sorted by predicted score."""
        scores = self.predict(features)
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -x[1])
        return [(cid, float(s)) for cid, s in ranked[:limit]]
