"""Scenario-family robustness and held-out random search over QLBS matrices."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import numpy as np

import _bootstrap
from benchmark.action_space import APPROVED_RESPONSES, IMPORTANT_PLAYER_ACTIONS, NPC_ACTIONS, PLAYER_ACTIONS
from benchmark.config import load_matrices, load_yaml
from src.qlbs import QLBS
from run_ngram_multiseed import predictability

ROOT = _bootstrap.ROOT


@dataclass
class Score:
    reaction: float
    k2: float


def perturb_matrices(base, rng, sigma=0.12):
    result = {}
    for action, matrix in base.items():
        noise = rng.lognormal(0.0, sigma, matrix.shape)
        result[action] = np.where(matrix > 0, matrix * noise, 0.0)
    return result


def perturb_profiles(base, rng, concentration=60.0):
    return {name: rng.dirichlet(concentration * values) for name, values in base.items()}


def reaction(players, npc):
    hits = total = 0
    for player, action in zip(players, npc):
        if player in IMPORTANT_PLAYER_ACTIONS:
            total += 1
            hits += NPC_ACTIONS[action] in APPROVED_RESPONSES[player]
    return hits / total if total else np.nan


def evaluate(matrices, profiles, seeds, episodes_per_profile, ticks, rerand=None):
    per_seed = []
    for seed in seeds:
        player_rng = np.random.default_rng(seed)
        controller_rng = np.random.default_rng(seed + 100_000)
        values = []
        for profile in profiles.values():
            for _ in range(episodes_per_profile):
                controller = QLBS(controller_rng, matrices, 0.75, 0.08, rerand)
                players, npc, seq = [], [], []
                for _ in range(ticks):
                    player_index = int(player_rng.choice(len(PLAYER_ACTIONS), p=profile))
                    player = PLAYER_ACTIONS[player_index]
                    action = controller.act(player)
                    players.append(player); npc.append(action); seq.append((player_index, action))
                values.append((reaction(players, npc), predictability(seq, 2)))
        per_seed.append(np.nanmean(values, axis=0))
    mean = np.mean(per_seed, axis=0)
    return Score(float(mean[0]), float(mean[1]))


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(quick=False):
    config = load_yaml(ROOT / "configs" / "benchmark.yaml")
    base_matrices = load_matrices(ROOT / "configs" / "matrices")
    base_profiles = {k: np.asarray(v, float) for k, v in config["profiles"].items()}
    scenario_count = 8 if quick else 30
    epp = 3 if quick else 8
    ticks = 100 if quick else 200
    scenario_rows = []
    for scenario in range(scenario_count):
        rng = np.random.default_rng(30_000 + scenario)
        matrices = perturb_matrices(base_matrices, rng)
        profiles = perturb_profiles(base_profiles, rng)
        plain = evaluate(matrices, profiles, [scenario], epp, ticks)
        rerand = evaluate(matrices, profiles, [scenario], epp, ticks, rerand=15)
        scenario_rows.append({
            "scenario": scenario,
            "plain_reaction": f"{plain.reaction:.6f}", "plain_k2": f"{plain.k2:.6f}",
            "rerand_reaction": f"{rerand.reaction:.6f}", "rerand_k2": f"{rerand.k2:.6f}",
            "rerand_reduces_k2": int(rerand.k2 < plain.k2),
        })
    write(ROOT / "results" / "generated" / "scenario_generalization.csv", scenario_rows)

    candidate_count = 12 if quick else 80
    train_profiles = base_profiles
    candidates = []
    for candidate in range(candidate_count):
        rng = np.random.default_rng(90_000 + candidate)
        matrices = perturb_matrices(base_matrices, rng, sigma=0.18)
        score = evaluate(matrices, train_profiles, [101, 103], epp, ticks)
        objective = score.reaction - 0.50 * score.k2
        candidates.append((objective, candidate, matrices, score))
    _, best_id, best_matrices, train_score = max(candidates, key=lambda item: item[0])
    test_seeds = [211, 223] if quick else [211, 223, 227, 229, 233, 239]
    test_epp = 4 if quick else 20
    base_test = evaluate(base_matrices, train_profiles, test_seeds, test_epp, 100 if quick else 300)
    best_test = evaluate(best_matrices, train_profiles, test_seeds, test_epp, 100 if quick else 300)
    search_rows = [
        {"matrix_set": "hand_tuned", "selected_candidate": "", "reaction": f"{base_test.reaction:.6f}", "k2": f"{base_test.k2:.6f}"},
        {"matrix_set": "random_search", "selected_candidate": best_id, "reaction": f"{best_test.reaction:.6f}", "k2": f"{best_test.k2:.6f}"},
    ]
    write(ROOT / "results" / "generated" / "matrix_search_heldout.csv", search_rows)
    np.savez(ROOT / "results" / "generated" / "learned_matrices.npz", **best_matrices)
    print(f"Generalization: rerandomization reduced k2 in {sum(int(r['rerand_reduces_k2']) for r in scenario_rows)}/{scenario_count} scenarios.")
    print(f"Matrix search selected candidate {best_id}; held-out base={base_test}, selected={best_test}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
