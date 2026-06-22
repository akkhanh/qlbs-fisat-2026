from __future__ import annotations

import time
from dataclasses import dataclass

from .environment import PlayerEnvironment
from .metrics import summarize_episode


@dataclass
class Episode:
    profile: str
    player_actions: list[str]
    npc_actions: list[int]
    decision_time_ms: float


def run_episode(controller, environment: PlayerEnvironment, profile: str, ticks: int) -> Episode:
    controller.reset()
    player_actions: list[str] = []
    npc_actions: list[int] = []
    elapsed_ns = 0
    for _ in range(ticks):
        player_action = environment.sample(profile)
        start = time.perf_counter_ns()
        npc_action = controller.act(player_action)
        elapsed_ns += time.perf_counter_ns() - start
        player_actions.append(player_action)
        npc_actions.append(npc_action)
    return Episode(profile, player_actions, npc_actions, elapsed_ns / ticks / 1e6)


def episode_row(model: str, seed: int, episode_id: int, episode: Episode) -> dict[str, object]:
    row = {
        "model": model,
        "seed": seed,
        "episode_id": episode_id,
        "profile": episode.profile,
        "decision_time_ms": episode.decision_time_ms,
    }
    row.update(summarize_episode(episode.player_actions, episode.npc_actions))
    return row
