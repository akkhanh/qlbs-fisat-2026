from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

import _bootstrap
from benchmark.action_space import PLAYER_ACTIONS, PLAYER_INDEX
from benchmark.config import load_matrices, load_yaml
from benchmark.environment import PlayerEnvironment
from benchmark.metrics import reaction_match
from src.qlbs import QLBS
from run_ngram_multiseed import predictability

ROOT = _bootstrap.ROOT


def evaluate(rho: float = 0.0, rerandomize_every: int | None = None, quick: bool = False):
    config = load_yaml(ROOT / "configs" / "benchmark.yaml")
    matrices = load_matrices(ROOT / "configs" / "matrices")
    profiles = {key: np.asarray(value, dtype=float) for key, value in config["profiles"].items()}
    seeds = range(2 if quick else 6)
    episodes_per_profile = 8 if quick else 40
    ticks = 100 if quick else 300
    seed_metrics = []
    for seed in seeds:
        player_rng = np.random.default_rng(seed)
        controller_rng = np.random.default_rng(seed + 10_000)
        environment = PlayerEnvironment(profiles, player_rng)
        values = []
        for profile in profiles:
            for _ in range(episodes_per_profile):
                controller = QLBS(
                    controller_rng, matrices, config["qlbs"]["gamma"],
                    config["qlbs"]["memory_decay"], rerandomize_every, rho,
                )
                players, npc, sequence = [], [], []
                for _ in range(ticks):
                    p = environment.sample(profile)
                    a = controller.act(p)
                    players.append(p); npc.append(a); sequence.append((PLAYER_INDEX[p], a))
                values.append((reaction_match(players, npc), predictability(sequence, 0), predictability(sequence, 2)))
        seed_metrics.append(np.nanmean(values, axis=0))
    array = np.asarray(seed_metrics)
    means = array.mean(axis=0)
    margins = 1.96 * array.std(axis=0, ddof=1) / np.sqrt(len(array))
    return means, margins


def run(quick: bool = False):
    rows = []
    for rho in (0.0, 0.10, 0.30, 0.50, 0.75, 1.0):
        means, cis = evaluate(rho=rho, quick=quick)
        rows.append(("anchor", rho, "", *means, *cis))
    means, cis = evaluate(rerandomize_every=15, quick=quick)
    rows.append(("rerandomization", 0.0, 15, *means, *cis))
    output = ROOT / "results" / "generated" / "qlbs_variant_tradeoff.csv"
    fields = ("variant", "rho", "rerandomize_every", "reaction_mean", "k0_mean", "k2_mean", "reaction_ci95", "k0_ci95", "k2_ci95")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
