from __future__ import annotations

import numpy as np

from .base import Controller


class MarkovBelief(Controller):
    """POMDP-lite posterior over player profiles."""

    name = "MarkovBelief"

    def __init__(
        self,
        rng: np.random.Generator,
        profile_action_likelihoods: np.ndarray,
        profile_response_policies: np.ndarray,
        smoothing: float = 0.03,
    ):
        super().__init__(rng)
        self.likelihoods = profile_action_likelihoods
        self.response_policies = profile_response_policies
        self.smoothing = smoothing
        self.reset()

    def reset(self) -> None:
        self.posterior = np.full(self.likelihoods.shape[0], 1.0 / self.likelihoods.shape[0])

    def policy(self, player_action: str) -> np.ndarray:
        from benchmark.action_space import PLAYER_INDEX

        action_index = PLAYER_INDEX[player_action]
        posterior = self.posterior * self.likelihoods[:, action_index]
        posterior /= posterior.sum()
        uniform = np.full(len(posterior), 1.0 / len(posterior))
        self.posterior = (1.0 - self.smoothing) * posterior + self.smoothing * uniform
        policy = self.posterior @ self.response_policies[:, action_index, :]
        return policy / policy.sum()
