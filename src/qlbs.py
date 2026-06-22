from __future__ import annotations

import numpy as np

from benchmark.action_space import NPC_ACTIONS, NPC_INDEX, IMPORTANT_PLAYER_ACTIONS, APPROVED_RESPONSES
from .base import Controller


class QLBS(Controller):
    name = "QLBS"

    def __init__(
        self,
        rng: np.random.Generator,
        matrices: dict[str, np.ndarray],
        gamma: float = 0.75,
        memory_decay: float = 0.08,
        rerandomize_every: int | None = None,
        anchor_probability: float = 0.0,
    ):
        super().__init__(rng)
        self.matrices = matrices
        self.gamma = gamma
        self.memory_decay = memory_decay
        self.rerandomize_every = rerandomize_every
        self.anchor_probability = anchor_probability
        self.reset()

    def reset(self) -> None:
        self.belief = np.full(len(NPC_ACTIONS), 1.0 / len(NPC_ACTIONS))
        self.tick = 0

    def policy(self, player_action: str) -> np.ndarray:
        if self.rerandomize_every and self.tick > 0 and self.tick % self.rerandomize_every == 0:
            self.belief[:] = 1.0 / len(self.belief)
        transformed = self.matrices[player_action] @ self.belief
        transformed /= transformed.sum()
        uniform = np.full(len(self.belief), 1.0 / len(self.belief))
        self.belief = (1.0 - self.memory_decay) * transformed + self.memory_decay * uniform
        weights = np.power(np.clip(self.belief, 1e-12, None), self.gamma)
        self.tick += 1
        return weights / weights.sum()

    def act(self, player_action: str) -> int:
        probabilities = self.policy(player_action)
        if (
            self.anchor_probability > 0
            and player_action in IMPORTANT_PLAYER_ACTIONS
            and self.rng.random() < self.anchor_probability
        ):
            approved = sorted(NPC_INDEX[action] for action in APPROVED_RESPONSES[player_action])
            return int(self.rng.choice(approved))
        return int(self.rng.choice(len(probabilities), p=probabilities))
