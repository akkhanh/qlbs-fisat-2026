from __future__ import annotations

import numpy as np

from .action_space import PLAYER_ACTIONS


class PlayerEnvironment:
    def __init__(self, profiles: dict[str, np.ndarray], rng: np.random.Generator):
        self.profiles = profiles
        self.rng = rng

    def sample(self, profile: str) -> str:
        index = int(self.rng.choice(len(PLAYER_ACTIONS), p=self.profiles[profile]))
        return PLAYER_ACTIONS[index]
