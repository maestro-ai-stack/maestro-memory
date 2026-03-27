"""River streaming pre-ranker for online learning."""
from __future__ import annotations


class OnlineRanker:
    """Streaming pre-ranker using River Hoeffding tree.
    Learns from implicit feedback (which facts were actually used)."""

    def __init__(self):
        self._model = None
        self._n_updates = 0

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from river import tree
            self._model = tree.HoeffdingTreeClassifier()
            return True
        except ImportError:
            return False

    def predict(self, features: dict) -> float:
        """Predict probability that this candidate will be used."""
        if not self._ensure_model():
            return 0.5
        proba = self._model.predict_proba_one(features)
        return proba.get(True, 0.5)

    def update(self, features: dict, used: bool) -> None:
        """Learn from one observation."""
        if not self._ensure_model():
            return
        self._model.learn_one(features, used)
        self._n_updates += 1

    @property
    def n_updates(self) -> int:
        return self._n_updates
