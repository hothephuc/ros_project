"""Step A5: how hard a push can H1 survive under each balance controller?

For each controller, push direction and force, push the pelvis for 0.1 s at a random time in
[2, 3] s, five times with different push times, and count how often H1 is still standing at 6 s.

    uv run python sim/push_experiment.py    # writes results/h1_push_survival.csv and .png
"""

from pathlib import Path

import mujoco
import numpy as np

from balance_h1 import FALLEN, AnkleBalance
from humanoid import Humanoid

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONTROLLERS = ["pitch", "com"]
DIRECTIONS = {"forward": 1.0, "backward": -1.0}
FORCES = np.arange(0, 401, 25)
REPEATS = 5
PUSH_FOR, SECONDS = 0.1, 6.0


def survives(mode: str, force: float, push_at: float) -> bool:
    robot = Humanoid("h1")
    balance = AnkleBalance(robot, mode)
    while robot.data.time < SECONDS:
        pushing = push_at <= robot.data.time < push_at + PUSH_FOR
        robot.push(np.array([force if pushing else 0.0, 0.0, 0.0]))
        balance.step()
        if robot.height < FALLEN:
            return False
    return True


def main() -> None:
    rng = np.random.default_rng(0)
    push_times = rng.uniform(2.0, 3.0, REPEATS)
    rows = []
    for mode in CONTROLLERS:
        for direction, sign in DIRECTIONS.items():
            rates = []
            for force in FORCES:
                ok = sum(survives(mode, sign * force, t) for t in push_times)
                rates.append(ok / REPEATS)
                rows.append((mode, direction, int(force), ok, REPEATS))
            print(f"{mode:>5} {direction:>8}: " + " ".join(f"{r:.1f}" for r in rates))
    RESULTS.mkdir(exist_ok=True)
    np.savetxt(
        RESULTS / "h1_push_survival.csv",
        np.array(rows, dtype=object),
        fmt="%s",
        delimiter=",",
        header="controller,direction,force_N,survived,trials",
        comments="",
    )

    # Largest force every trial survived, per controller and direction.
    for mode in CONTROLLERS:
        for direction in DIRECTIONS:
            safe = [f for m, d, f, ok, n in rows if m == mode and d == direction and ok == n]
            first_fail = next((f for m, d, f, ok, n in rows if m == mode and d == direction and ok < n), None)
            print(f"{mode:>5} {direction:>8}: always survives up to {max(safe)} N; first failure at {first_fail} N")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, color in zip(CONTROLLERS, ["tab:gray", "tab:blue"]):
        for direction, style in zip(DIRECTIONS, ["-", "--"]):
            rate = [ok / n for m, d, f, ok, n in rows if m == mode and d == direction]
            ax.plot(FORCES, rate, style, color=color, marker="o", label=f"{mode} feedback, {direction} push")
    ax.set_xlabel("push force on the pelvis for 0.1 s [N]")
    ax.set_ylabel(f"survival rate ({REPEATS} trials)")
    ax.set_title("H1 push recovery: CoM feedback vs torso-pitch feedback")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "h1_push_survival.png", dpi=150)
    print(f"saved {RESULTS / 'h1_push_survival.png'}")


if __name__ == "__main__":
    main()
