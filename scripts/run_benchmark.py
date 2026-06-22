from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import numpy as np

import _bootstrap
from benchmark.action_space import PLAYER_INDEX
from benchmark.config import load_matrices, load_yaml
from benchmark.environment import PlayerEnvironment
from benchmark.factory import build_controller
from benchmark.simulator import episode_row, run_episode


ROOT = _bootstrap.ROOT


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(1.96 * np.nanstd(array, ddof=1) / np.sqrt(np.sum(~np.isnan(array))))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(quick: bool = False) -> None:
    config = load_yaml(ROOT / "configs" / "benchmark.yaml")
    if quick:
        config.update(load_yaml(ROOT / "configs" / "quick.yaml"))
    matrices = load_matrices(ROOT / "configs" / "matrices")
    profiles = {key: np.asarray(value, dtype=float) for key, value in config["profiles"].items()}
    summary_rows: list[dict] = []
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    for model_index, model_name in enumerate(config["models"]):
        seed = int(config["seed"]) + model_index * 100_003
        player_rng = np.random.default_rng(seed)
        controller_rng = np.random.default_rng(seed + 1)
        environment = PlayerEnvironment(profiles, player_rng)
        controller = build_controller(model_name, controller_rng, config, matrices)
        episode_id = 0
        log_path = log_dir / f"{slug(model_name)}.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as log_handle:
            log = csv.writer(log_handle)
            log.writerow(("episode_id", "tick", "player_action", "npc_action", "profile"))
            for profile in profiles:
                for _ in range(int(config["episodes_per_profile"])):
                    episode = run_episode(controller, environment, profile, int(config["ticks_per_episode"]))
                    summary_rows.append(episode_row(model_name, seed, episode_id, episode))
                    for tick, (player_action, npc_action) in enumerate(zip(episode.player_actions, episode.npc_actions)):
                        log.writerow((episode_id, tick, PLAYER_INDEX[player_action], npc_action, profile))
                    episode_id += 1

    write_rows(ROOT / "results" / "generated" / "episode_summary.csv", summary_rows)
    metrics = ("coverage", "transitions", "entropy", "predictability", "adaptation_shift", "reaction_match", "decision_time_ms")
    table = []
    for model_name in config["models"]:
        selected = [row for row in summary_rows if row["model"] == model_name]
        output = {"model": model_name}
        for metric in metrics:
            mean, margin = ci([float(row[metric]) for row in selected])
            output[f"{metric}_mean"] = f"{mean:.6f}"
            output[f"{metric}_ci95"] = f"{margin:.6f}"
        table.append(output)
    write_rows(ROOT / "results" / "generated" / "benchmark_summary.csv", table)
    print(f"Wrote {len(summary_rows)} episode summaries and {len(config['models'])} controller logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run the small smoke-test configuration.")
    run(parser.parse_args().quick)
