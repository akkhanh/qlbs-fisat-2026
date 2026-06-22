"""Dependency-light learned adversary for every controller log."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

import _bootstrap

ROOT = _bootstrap.ROOT
N_PLAYER, N_NPC, CONTEXT = 5, 6, 8


def load_episodes(path: Path):
    episodes = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episodes[int(row["episode_id"])].append((int(row["player_action"]), int(row["npc_action"])))
    return episodes


def one_hot(value: int, size: int):
    result = np.zeros(size, dtype=np.float32)
    result[value] = 1.0
    return result


def build_xy(episodes, ids):
    features, targets = [], []
    for episode_id in ids:
        sequence = episodes[episode_id]
        for t, (current_player, current_npc) in enumerate(sequence):
            context = []
            for lag in range(CONTEXT, 0, -1):
                if t - lag >= 0:
                    player, npc = sequence[t - lag]
                else:
                    player, npc = N_PLAYER, N_NPC
                context.extend((one_hot(player, N_PLAYER + 1), one_hot(npc, N_NPC + 1)))
            context.append(one_hot(current_player, N_PLAYER + 1))
            features.append(np.concatenate(context))
            targets.append(current_npc)
    return np.asarray(features), np.asarray(targets)


def softmax(values):
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def train_eval(episodes, seed: int, epochs: int):
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(episodes))
    rng.shuffle(ids)
    split = int(0.8 * len(ids))
    x_train, y_train = build_xy(episodes, ids[:split])
    x_test, y_test = build_xy(episodes, ids[split:])
    hidden_size = 64
    w1 = rng.normal(0, 0.01, (x_train.shape[1], hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0, 0.01, (hidden_size, N_NPC)).astype(np.float32)
    b2 = np.zeros(N_NPC, dtype=np.float32)
    for _ in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), 512):
            batch = order[start:start + 512]
            x, y = x_train[batch], y_train[batch]
            hidden = np.maximum(0, x @ w1 + b1)
            probabilities = softmax(hidden @ w2 + b2)
            gradient = probabilities
            gradient[np.arange(len(y)), y] -= 1
            gradient /= len(y)
            grad_w2 = hidden.T @ gradient
            grad_b2 = gradient.sum(axis=0)
            grad_hidden = gradient @ w2.T
            grad_hidden[hidden <= 0] = 0
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            w1 -= 0.005 * grad_w1; b1 -= 0.005 * grad_b1
            w2 -= 0.005 * grad_w2; b2 -= 0.005 * grad_b2
    probabilities = softmax(np.maximum(0, x_test @ w1 + b1) @ w2 + b2)
    accuracy = float(np.mean(probabilities.argmax(axis=1) == y_test))
    cross_entropy = float(-np.log(probabilities[np.arange(len(y_test)), y_test] + 1e-12).mean())
    return accuracy, cross_entropy


def run(quick: bool = False):
    rows = []
    seeds = range(2 if quick else 5)
    epochs = 2 if quick else 8
    for path in sorted((ROOT / "logs").glob("*.csv")):
        episodes = load_episodes(path)
        results = np.asarray([train_eval(episodes, seed, epochs) for seed in seeds])
        rows.append({
            "model": path.stem,
            "top1_mean": f"{results[:, 0].mean():.6f}",
            "top1_ci95": f"{1.96 * results[:, 0].std(ddof=1) / np.sqrt(len(results)):.6f}" if len(results) > 1 else "nan",
            "cross_entropy_mean": f"{results[:, 1].mean():.6f}",
            "seeds": len(results),
        })
    output = ROOT / "results" / "generated" / "learned_adversary.csv"
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    run(parser.parse_args().quick)
