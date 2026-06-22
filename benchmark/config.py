from __future__ import annotations

import csv
from pathlib import Path
import yaml
import numpy as np

from benchmark.action_space import NPC_ACTIONS, PLAYER_ACTIONS


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_matrices(directory: str | Path) -> dict[str, np.ndarray]:
    directory = Path(directory)
    matrices = {}
    for player_action in PLAYER_ACTIONS:
        path = directory / f"U_{player_action.lower()}.csv"
        matrices[player_action] = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, len(NPC_ACTIONS) + 1))
    return matrices


def array_mapping(mapping: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in mapping.items()}
