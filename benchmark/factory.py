from __future__ import annotations

import numpy as np

from src import (
    BehaviorTree, FSM, MarkovBelief, QLBS, SoftmaxUtilityAI,
    StochasticBehaviorTree, UtilityAI,
)
from .action_space import NPC_INDEX


def build_controller(name: str, rng: np.random.Generator, config: dict, matrices: dict[str, np.ndarray]):
    scores = {key: np.asarray(value, dtype=float) for key, value in config["utility_scores"].items()}
    bt_policies = {key: np.asarray(value, dtype=float) for key, value in config["bt_policies"].items()}
    if name == "FSM":
        transitions = {action: NPC_INDEX[target] for action, target in config["fsm_transitions"].items()}
        return FSM(rng, transitions)
    if name == "Behavior Tree":
        return BehaviorTree(rng, bt_policies)
    if name == "Stochastic BT":
        return StochasticBehaviorTree(rng, bt_policies)
    if name == "Utility AI":
        return UtilityAI(rng, scores)
    if name == "Softmax Utility AI":
        return SoftmaxUtilityAI(rng, scores, config["softmax_temperature"])
    if name == "MarkovBelief":
        likelihoods = np.asarray(list(config["profiles"].values()), dtype=float)
        responses = np.asarray(config["markov_response_policies"], dtype=float)
        return MarkovBelief(rng, likelihoods, responses, config["markov_smoothing"])
    if name == "QLBS":
        q = config["qlbs"]
        return QLBS(rng, matrices, q["gamma"], q["memory_decay"])
    raise KeyError(name)
