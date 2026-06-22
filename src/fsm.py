from __future__ import annotations

import numpy as np

from .base import Controller


class FSM(Controller):
    name = "FSM"

    def __init__(self, rng: np.random.Generator, transitions: dict[str, int], neutral_action: int = 0):
        super().__init__(rng)
        self.transitions = transitions
        self.neutral_action = neutral_action
        self.reset()

    def reset(self) -> None:
        self.state = self.neutral_action

    def policy(self, player_action: str) -> np.ndarray:
        self.state = self.transitions.get(player_action, self.state)
        result = np.zeros(6)
        result[self.state] = 1.0
        return result
