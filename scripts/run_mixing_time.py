from __future__ import annotations

import argparse
import csv
import numpy as np

import _bootstrap
from benchmark.action_space import PLAYER_ACTIONS, PLAYER_INDEX
from benchmark.config import load_matrices, load_yaml
from benchmark.environment import PlayerEnvironment
from benchmark.metrics import js_distance
from src.qlbs import QLBS
from run_ngram_multiseed import predictability

ROOT = _bootstrap.ROOT


def cps(matrices, gamma: float, decay: float, phase_len: int = 40) -> float:
    scripted = QLBS(np.random.default_rng(1), matrices, gamma, decay)
    neutral = QLBS(np.random.default_rng(2), matrices, gamma, decay)
    script = (["Steal", "Hide"] * phase_len)[:phase_len] + ["Move"] * phase_len + ["Talk"] * phase_len
    distances = []
    for player_action in script:
        distances.append(js_distance(scripted.policy(player_action), neutral.policy("Move")))
    return float(np.mean(distances))


def heldout_k2(matrices, config, decay: float, quick: bool) -> tuple[float, float]:
    profiles = {key: np.asarray(value, dtype=float) for key, value in config["profiles"].items()}
    seed_scores = []
    for seed in range(2 if quick else 6):
        environment = PlayerEnvironment(profiles, np.random.default_rng(seed))
        controller_rng = np.random.default_rng(seed + 20_000)
        scores = []
        for profile in profiles:
            for _ in range(8 if quick else 40):
                controller = QLBS(controller_rng, matrices, config["qlbs"]["gamma"], decay)
                sequence = []
                for _ in range(100 if quick else 300):
                    player_action = environment.sample(profile)
                    sequence.append((PLAYER_INDEX[player_action], controller.act(player_action)))
                scores.append(predictability(sequence, 2))
        seed_scores.append(np.nanmean(scores))
    values = np.asarray(seed_scores)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def run(quick: bool = False):
    config = load_yaml(ROOT / "configs" / "benchmark.yaml")
    matrices = load_matrices(ROOT / "configs" / "matrices")
    rows = []
    for decay in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25):
        pred, ci = heldout_k2(matrices, config, decay, quick)
        rows.append((decay, cps(matrices, config["qlbs"]["gamma"], decay), pred, ci))
    output = ROOT / "results" / "generated" / "mixing_tradeoff.csv"
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("memory_decay", "cps", "predictability_k2_mean", "predictability_k2_ci95"))
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
