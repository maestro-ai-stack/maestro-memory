"""Thompson Sampling blender for channel weight optimization."""
from __future__ import annotations

import numpy as np


class ThompsonBlender:
    """Thompson Sampling for recall channel blending weights.

    Each channel has a Beta(alpha, beta) prior. On each search,
    sample weights from the posterior. Update based on whether
    the channel contributed useful results.
    """

    CHANNEL_NAMES = [
        "bm25",
        "embedding",
        "graph",
        "user_interest",
        "time_window",
        "session_context",
    ]

    def __init__(self, n_channels: int = 6):
        self.alpha = np.ones(n_channels, dtype=np.float64)
        self.beta = np.ones(n_channels, dtype=np.float64)
        self._n_updates = 0

    def sample_weights(self) -> np.ndarray:
        """Sample channel weights from Beta posteriors."""
        weights = np.random.beta(self.alpha, self.beta)
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def update(self, channel_idx: int, reward: float) -> None:
        """Update posterior for a channel based on reward (0-1)."""
        if reward > 0.5:
            self.alpha[channel_idx] += 1
        else:
            self.beta[channel_idx] += 1
        self._n_updates += 1

    def get_stats(self) -> dict:
        """Return channel statistics."""
        means = self.alpha / (self.alpha + self.beta)
        return {
            name: {
                "mean": float(means[i]),
                "alpha": float(self.alpha[i]),
                "beta": float(self.beta[i]),
            }
            for i, name in enumerate(self.CHANNEL_NAMES[: len(self.alpha)])
        }

    @property
    def n_updates(self) -> int:
        return self._n_updates
