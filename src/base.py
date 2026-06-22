from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """Common controller interface used by the benchmark."""

    name = "Controller"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def policy(self, player_action: str) -> np.ndarray:
        """Return a normalized NPC-action distribution."""

    def act(self, player_action: str) -> int:
        probabilities = self.policy(player_action)
        return int(self.rng.choice(len(probabilities), p=probabilities))
