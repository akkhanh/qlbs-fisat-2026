from __future__ import annotations

import numpy as np

from .utility_ai import UtilityAI


class SoftmaxUtilityAI(UtilityAI):
    name = "Softmax Utility AI"

    def __init__(self, rng: np.random.Generator, scores: dict[str, np.ndarray], temperature: float = 0.20):
        self.temperature = temperature
        super().__init__(rng, scores)

    def policy(self, player_action: str) -> np.ndarray:
        scores = self.scores[player_action] / self.temperature
        weights = np.exp(scores - scores.max())
        return weights / weights.sum()
