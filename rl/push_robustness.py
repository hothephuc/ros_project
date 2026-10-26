"""Step A5 (RL option): do policies trained with random pushes survive bigger pushes?

Evaluates each run under pushes of fixed magnitude (random sign and timing), 5 episodes per force,
and reports the mean return and how many episodes reached the full 1,000 steps.

    uv run python rl/push_robustness.py ppo_5m_norm ppo_5m_push   # writes results/walker_push_robustness.csv/.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate import evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
FORCES = [0, 50, 100, 150, 200, 250, 300]
EPISODES = 5
FULL = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    rows = []
    for name in args.runs:
        for force in FORCES:
            results = evaluate(name, EPISODES, float(force))
            returns = np.array([r for r, _ in results])
            survived = sum(n == FULL for _, n in results)
            rows.append((name, force, round(returns.mean()), round(returns.std()), survived, EPISODES))
            print(f"{name:>14} {force:>4} N: return {returns.mean():>6,.0f} ± {returns.std():>5,.0f}, "
                  f"{survived}/{EPISODES} full episodes")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "walker_push_robustness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "push_N", "mean_return", "std_return", "full_episodes", "episodes"])
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in args.runs:
        mine = [r for r in rows if r[0] == name]
        ax.errorbar([r[1] for r in mine], [r[2] for r in mine], yerr=[r[3] for r in mine],
                    marker="o", capsize=3, label=name)
    ax.set_xlabel("push on the torso during evaluation [N] (40 ms, random sign, ~1% of steps)")
    ax.set_ylabel(f"mean return over {EPISODES} episodes")
    ax.set_title("Walker2d: does training with pushes make the policy more robust?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "walker_push_robustness.png", dpi=150)
    print(f"saved {RESULTS / 'walker_push_robustness.png'}")


if __name__ == "__main__":
    main()
