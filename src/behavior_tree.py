from __future__ import annotations

import numpy as np

from .base import Controller


class BehaviorTree(Controller):
    name = "Behavior Tree"

    def __init__(self, rng: np.random.Generator, policies: dict[str, np.ndarray]):
        super().__init__(rng)
        self.policies = policies
        self.reset()

    def reset(self) -> None:
        pass

    def policy(self, player_action: str) -> np.ndarray:
        probabilities = self.policies[player_action]
        result = np.zeros_like(probabilities)
        result[int(np.argmax(probabilities))] = 1.0
        return result


class StochasticBehaviorTree(BehaviorTree):
    name = "Stochastic BT"

    def policy(self, player_action: str) -> np.ndarray:
        return self.policies[player_action].copy()
