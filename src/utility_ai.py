from __future__ import annotations

import numpy as np

from .base import Controller


class UtilityAI(Controller):
    name = "Utility AI"

    def __init__(self, rng: np.random.Generator, scores: dict[str, np.ndarray]):
        super().__init__(rng)
        self.scores = scores
        self.reset()

    def reset(self) -> None:
        pass

    def policy(self, player_action: str) -> np.ndarray:
        scores = self.scores[player_action]
        result = np.zeros_like(scores, dtype=float)
        result[int(np.argmax(scores))] = 1.0
        return result
