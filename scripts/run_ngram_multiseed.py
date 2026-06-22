from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import numpy as np

import _bootstrap
from benchmark.action_space import PLAYER_INDEX
from benchmark.config import load_matrices, load_yaml
from benchmark.environment import PlayerEnvironment
from benchmark.factory import build_controller
from benchmark.simulator import run_episode

ROOT = _bootstrap.ROOT


def predictability(sequence: list[tuple[int, int]], order: int) -> float:
    """Held-out majority n-gram accuracy on the second half of one episode."""
    half = len(sequence) // 2

    def context(t: int) -> tuple[int, ...]:
        values = [sequence[t][0]]
        for lag in range(1, order + 1):
            values.extend(sequence[t - lag] if t - lag >= 0 else (-1, -1))
        return tuple(values)

    counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for t in range(order, half):
        counts[context(t)][sequence[t][1]] += 1
    majority = {key: value.most_common(1)[0][0] for key, value in counts.items()}
    hits = [
        majority[context(t)] == sequence[t][1]
        for t in range(max(half, order), len(sequence))
        if context(t) in majority
    ]
    return float(np.mean(hits)) if hits else float("nan")


def run(quick: bool = False) -> None:
    config = load_yaml(ROOT / "configs" / "benchmark.yaml")
    matrices = load_matrices(ROOT / "configs" / "matrices")
    profiles = {key: np.asarray(value, dtype=float) for key, value in config["profiles"].items()}
    seeds = [11, 23] if quick else [11, 23, 37, 53, 71, 97]
    episodes_per_profile = 4 if quick else 40
    ticks = 100 if quick else int(config["ticks_per_episode"])
    per_seed: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for model_index, model_name in enumerate(config["models"]):
        for seed in seeds:
            model_seed = seed + model_index * 100_003
            environment = PlayerEnvironment(profiles, np.random.default_rng(model_seed))
            controller = build_controller(
                model_name, np.random.default_rng(model_seed + 1), config, matrices
            )
            for profile in profiles:
                for _ in range(episodes_per_profile):
                    episode = run_episode(controller, environment, profile, ticks)
                    sequence = [
                        (PLAYER_INDEX[player], npc)
                        for player, npc in zip(episode.player_actions, episode.npc_actions)
                    ]
                    for order in range(4):
                        per_seed[(model_name, seed, order)].append(predictability(sequence, order))

    rows = []
    for model_name in config["models"]:
        for order in range(4):
            seed_scores = np.asarray([
                np.nanmean(per_seed[(model_name, seed, order)]) for seed in seeds
            ])
            rows.append({
                "model": model_name,
                "order": order,
                "mean": f"{seed_scores.mean():.6f}",
                "ci95": f"{1.96 * seed_scores.std(ddof=1) / np.sqrt(len(seed_scores)):.6f}",
                "seeds": len(seeds),
                "episodes_per_profile_per_seed": episodes_per_profile,
            })

    output = ROOT / "results" / "generated" / "ngram_predictability_multiseed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} from {len(seeds)} independent seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use two short smoke-test seeds.")
    run(parser.parse_args().quick)
