from __future__ import annotations

import math
import numpy as np

from .action_space import IMPORTANT_PLAYER_ACTIONS, APPROVED_RESPONSES, NPC_ACTIONS


def action_distribution(actions: list[int]) -> np.ndarray:
    counts = np.bincount(actions, minlength=len(NPC_ACTIONS)).astype(float)
    return counts / counts.sum()


def coverage(actions: list[int]) -> float:
    return len(set(actions)) / len(NPC_ACTIONS)


def transition_variety(actions: list[int]) -> int:
    return len(set(zip(actions[:-1], actions[1:])))


def entropy(actions: list[int]) -> float:
    probabilities = action_distribution(actions)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def predictability(player_actions: list[str], npc_actions: list[int]) -> float:
    correct = 0
    for player_action in sorted(set(player_actions)):
        indices = [i for i, action in enumerate(player_actions) if action == player_action]
        values = [npc_actions[i] for i in indices]
        majority = int(np.argmax(np.bincount(values, minlength=len(NPC_ACTIONS))))
        correct += sum(npc_actions[i] == majority for i in indices)
    return correct / len(npc_actions)


def reaction_match(player_actions: list[str], npc_actions: list[int]) -> float:
    important = [i for i, action in enumerate(player_actions) if action in IMPORTANT_PLAYER_ACTIONS]
    if not important:
        return float("nan")
    return sum(NPC_ACTIONS[npc_actions[i]] in APPROVED_RESPONSES[player_actions[i]] for i in important) / len(important)


def js_distance(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    mask_left = left > 0
    mask_right = right > 0
    kl_left = np.sum(left[mask_left] * np.log2(left[mask_left] / midpoint[mask_left]))
    kl_right = np.sum(right[mask_right] * np.log2(right[mask_right] / midpoint[mask_right]))
    return float(math.sqrt(max(0.5 * (kl_left + kl_right), 0.0)))


def adaptation_shift(player_actions: list[str], npc_actions: list[int], window: int = 8) -> float:
    shifts = []
    for i, action in enumerate(player_actions):
        if action not in IMPORTANT_PLAYER_ACTIONS or i < window or i + window >= len(npc_actions):
            continue
        before = action_distribution(npc_actions[i - window:i])
        after = action_distribution(npc_actions[i + 1:i + 1 + window])
        shifts.append(js_distance(before, after))
    return float(np.mean(shifts)) if shifts else float("nan")


def summarize_episode(player_actions: list[str], npc_actions: list[int]) -> dict[str, float]:
    return {
        "coverage": coverage(npc_actions),
        "transitions": transition_variety(npc_actions),
        "entropy": entropy(npc_actions),
        "predictability": predictability(player_actions, npc_actions),
        "adaptation_shift": adaptation_shift(player_actions, npc_actions),
        "reaction_match": reaction_match(player_actions, npc_actions),
    }
