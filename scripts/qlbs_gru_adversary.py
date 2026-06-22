"""Optional PyTorch GRU adversary. Install requirements-gru.txt first."""
from __future__ import annotations

import glob
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N_PLAYER, N_NPC, CONTEXT = 5, 6, 8


def build_xy(frame):
    features, targets = [], []
    for _, group in frame.sort_values(["episode_id", "tick"]).groupby("episode_id"):
        players, npc = group.player_action.values, group.npc_action.values
        for t in range(len(players)):
            context = []
            for lag in range(CONTEXT, 0, -1):
                context.extend((players[t-lag], npc[t-lag]) if t-lag >= 0 else (N_PLAYER, N_NPC))
            context.append(players[t])
            features.append(context); targets.append(npc[t])
    return np.asarray(features), np.asarray(targets)


class GRUAdversary(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.player_embedding = nn.Embedding(N_PLAYER + 1, 16)
        self.npc_embedding = nn.Embedding(N_NPC + 1, 16)
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 16, 64), nn.ReLU(), nn.Linear(64, N_NPC))

    def forward(self, values):
        pairs = values[:, :-1].reshape(values.size(0), CONTEXT, 2)
        embedded = torch.cat((self.player_embedding(pairs[..., 0]), self.npc_embedding(pairs[..., 1])), dim=-1)
        _, hidden = self.gru(embedded)
        return self.head(torch.cat((hidden[-1], self.player_embedding(values[:, -1])), dim=-1))


def evaluate(path, seed=0, epochs=12):
    torch.manual_seed(seed); np.random.seed(seed)
    frame = pd.read_csv(path)
    episode_ids = frame.episode_id.unique(); np.random.shuffle(episode_ids)
    cut = int(0.8 * len(episode_ids))
    train_ids, test_ids = set(episode_ids[:cut]), set(episode_ids[cut:])
    x_train, y_train = build_xy(frame[frame.episode_id.isin(train_ids)])
    x_test, y_test = build_xy(frame[frame.episode_id.isin(test_ids)])
    model = GRUAdversary(); optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=512, shuffle=True)
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(); loss = loss_function(model(x_batch), y_batch)
            loss.backward(); optimizer.step()
    with torch.no_grad():
        logits = model(torch.tensor(x_test))
        accuracy = (logits.argmax(1).numpy() == y_test).mean()
        cross_entropy = loss_function(logits, torch.tensor(y_test)).item()
    return float(accuracy), float(cross_entropy)


if __name__ == "__main__":
    for path in (sys.argv[1:] or sorted(glob.glob("logs/*.csv"))):
        results = np.asarray([evaluate(path, seed=seed) for seed in range(5)])
        ci = 1.96 * results[:, 0].std(ddof=1) / np.sqrt(len(results))
        print(f"{path}: top1={results[:,0].mean():.3f} +/- {ci:.3f}; xent={results[:,1].mean():.3f}")
