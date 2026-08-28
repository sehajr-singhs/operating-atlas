"""Run the fresh IOO benchmark and generate its figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ioo_experiment import make_episodes, run
from .ioo_core import trajectory_chart


def figures(output: str, steps: int = 800) -> None:
    out = Path(output)
    episodes = make_episodes(4, 1, steps)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ep in episodes:
        z = trajectory_chart(ep.telemetry[::4], 3)
        ax.plot(z[:, 0], z[:, 1], lw=0.8, alpha=0.7, label=f"M{ep.machine}")
    ax.set_title("IOO operational trajectories in a 3D chart, shown in 2D")
    ax.set_xlabel("chart coordinate 1")
    ax.set_ylabel("chart coordinate 2")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_manifold.png", dpi=180)
    plt.close(fig)

    result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    rows = result["rows"]
    names = sorted({r["representation"] for r in rows})
    values = {n: [r["balanced_accuracy"] for r in rows if r["representation"] == n] for n in names}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([values[n] for n in names], labels=names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Fresh IOO representation benchmark")
    fig.tight_layout()
    fig.savefig(out / "figure_transfer.png", dpi=180)
    plt.close(fig)

    grouped = [(n, np.mean(values[n])) for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([x[0] for x in grouped], [x[1] for x in grouped], color=["#687c97", "#4b8f8c", "#c05b5b"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean balanced accuracy")
    ax.set_title("Representation ablation")
    fig.tight_layout()
    fig.savefig(out / "figure_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/local")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--machines", type=int, default=8)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    run(args.output, args.seeds, args.machines, args.episodes, args.steps)
    figures(args.output, args.steps)
    print(f"wrote fresh IOO artifacts to {args.output}")


if __name__ == "__main__":
    main()
