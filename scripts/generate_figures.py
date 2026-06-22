from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap

ROOT = _bootstrap.ROOT


def model_key(name: str) -> str:
    return "_".join(name.lower().split())


def rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mixing():
    data = rows(ROOT / "results" / "generated" / "mixing_tradeoff.csv")
    x = [float(row["memory_decay"]) for row in data]
    cps = [float(row["cps"]) for row in data]
    pred = [float(row["predictability_k2_mean"]) for row in data]
    fig, left = plt.subplots(figsize=(5.4, 3.6))
    left.plot(x, cps, "o-", color="#0072B2")
    left.set_xlabel(r"memory decay $\lambda$")
    left.set_ylabel("context persistence (CPS)", color="#0072B2")
    right = left.twinx()
    right.plot(x, pred, "s--", color="#D55E00")
    right.set_ylabel("predictability at k=2", color="#D55E00")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "mixing_tradeoff.pdf")
    fig.savefig(ROOT / "figures" / "mixing_tradeoff.png", dpi=160)
    plt.close(fig)


def pareto():
    summary = {row["model"]: row for row in rows(ROOT / "results" / "generated" / "benchmark_summary.csv")}
    ngram = rows(ROOT / "results" / "generated" / "ngram_predictability_multiseed.csv")
    k2 = {model_key(row["model"]): float(row["mean"]) for row in ngram if int(row["order"]) == 2}
    variants = rows(ROOT / "results" / "generated" / "qlbs_variant_tradeoff.csv")
    label_offsets = {
        "FSM": (-8, 8),
        "Behavior Tree": (-8, -12),
        "Utility AI": (-8, 18),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for axis, order in zip(axes, (0, 2)):
        for model, row in summary.items():
            model_slug = model_key(model)
            y = float(row["predictability_mean"]) if order == 0 else k2.get(model_slug)
            if y is None:
                continue
            x = float(row["reaction_match_mean"])
            axis.scatter(x, y, s=45)
            axis.annotate(
                model, (x, y), fontsize=6,
                xytext=label_offsets.get(model, (3, 3)), textcoords="offset points",
            )
        shown = set()
        for row in variants:
            y = float(row["k0_mean"] if order == 0 else row["k2_mean"])
            x = float(row["reaction_mean"])
            variant = row["variant"]
            label = None
            if variant not in shown:
                label = "QLBS anchor sweep" if variant == "anchor" else "QLBS re-randomization"
                shown.add(variant)
            axis.scatter(
                x, y, marker="D" if variant == "anchor" else "^",
                color="#9467BD", label=label,
            )
        axis.set_xlabel("reaction match (higher is better)")
        axis.set_ylabel(f"predictability k={order} (lower is better)")
        axis.set_ylim(0.22, 1.09)
        axis.set_title("Memoryless observer" if order == 0 else "Memory-capable observer")
        axis.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pareto_predictability_reaction.pdf")
    fig.savefig(ROOT / "figures" / "pareto_predictability_reaction.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    mixing()
    pareto()
    print("Wrote figures/*.pdf and figures/*.png")
