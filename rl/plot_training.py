"""Plot ep_rew_mean against training steps for every run in runs/, saved to results/walker_training.png.

    uv run python rl/plot_training.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS, RESULTS = ROOT / "runs", ROOT / "results"


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for progress in sorted(RUNS.glob("*/progress.csv")):
        with open(progress) as f:
            rows = [r for r in csv.DictReader(f) if r.get("rollout/ep_rew_mean")]
        steps = [int(r["time/total_timesteps"]) for r in rows]
        reward = [float(r["rollout/ep_rew_mean"]) for r in rows]
        ax.plot(steps, reward, label=f"{progress.parent.name} (final {reward[-1]:,.0f})")
    ax.set_xlabel("training steps")
    ax.set_ylabel("mean episode reward (ep_rew_mean)")
    ax.set_title("PPO on Walker2d-v5")
    ax.legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "walker_training.png", dpi=150)
    print(f"saved {RESULTS / 'walker_training.png'}")


if __name__ == "__main__":
    main()
