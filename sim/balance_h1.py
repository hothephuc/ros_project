"""Step A2: keep H1 upright with ankle-strategy balance feedback, and sweep its gains.

Joint PD alone keeps H1's limbs in shape, but nothing acts on the floating base, so it tips over
backward: its center of mass (CoM) starts only ~1 cm in front of the heel. Two ankle strategies:

  pitch  ankle offset = k1 * torso pitch + k2 * pitch rate          (the guide's version; fragile)
  com    ankle offset = k1 * (CoM x - foot centre x) + k2 * CoM vx  (keeps the CoM over the feet; robust)

    uv run python sim/balance_h1.py                    # H1 with no feedback, pitch and com feedback
    uv run python sim/balance_h1.py --sweep pitch      # gain grid, writes results/h1_sweep_pitch.csv
    uv run python sim/balance_h1.py --sweep com        # gain grid, writes results/h1_sweep_com.csv
    uv run mjpython sim/balance_h1.py --view com       # watch it balance; push it with Ctrl + right-drag
"""

import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from humanoid import Humanoid

RESULTS = Path(__file__).resolve().parents[1] / "results"
GAINS = {"none": (0.0, 0.0), "pitch": (5.0, 0.8), "com": (5.0, 2.0)}
SWEEP = {
    "pitch": ([-5.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0], [-1.0, 0.0, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5]),
    "com": ([-10.0, -5.0, 0.0, 3.0, 5.0, 7.0, 10.0, 15.0], [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]),
}
FOOT_CENTRE = 0.09  # x of the middle of H1's sole, relative to the ankle joint [m]
FALLEN = 0.5  # base height [m] below which we call it a fall


class AnkleBalance:
    """Ankle-strategy feedback on top of the keyframe PD controller."""

    def __init__(self, robot: Humanoid, mode: str = "com", k1: float | None = None, k2: float | None = None) -> None:
        self.robot, self.mode = robot, mode
        default_k1, default_k2 = GAINS[mode]
        self.k1 = default_k1 if k1 is None else k1
        self.k2 = default_k2 if k2 is None else k2
        self.ankles = [robot.actuator("left_ankle"), robot.actuator("right_ankle")]
        self.foot = robot.model.body("left_ankle_link").id

    def error(self) -> tuple[float, float]:
        """The quantity to drive to zero, and its rate."""
        robot = self.robot
        if self.mode == "pitch":
            return robot.pitch, robot.pitch_rate
        mujoco.mj_subtreeVel(robot.model, robot.data)
        com_x = robot.data.subtree_com[robot.pelvis, 0] - robot.data.xpos[self.foot, 0]
        return com_x - FOOT_CENTRE, float(robot.data.subtree_linvel[robot.pelvis, 0])

    def step(self) -> None:
        offsets = np.zeros(self.robot.model.nu)
        if self.mode != "none":
            e, de = self.error()
            offsets[self.ankles] = self.k1 * e + self.k2 * de
        self.robot.control(offsets)
        mujoco.mj_step(self.robot.model, self.robot.data)


def final_height(mode: str, k1: float | None = None, k2: float | None = None, seconds: float = 10.0) -> float:
    robot = Humanoid("h1")
    balance = AnkleBalance(robot, mode, k1, k2)
    while robot.data.time < seconds:
        balance.step()
    return robot.height


def sweep(mode: str) -> None:
    k1s, k2s = SWEEP[mode]
    RESULTS.mkdir(exist_ok=True)
    rows = []
    print(f"[{mode}] final base height [m] after 10 s; rows k1, columns k2")
    print("  k1 \\ k2 " + "".join(f"{k2:>7.1f}" for k2 in k2s))
    for k1 in k1s:
        heights = [final_height(mode, k1, k2) for k2 in k2s]
        rows += [(k1, k2, round(h, 3), h > FALLEN) for k2, h in zip(k2s, heights)]
        print(f"  {k1:>7.1f} " + "".join(f"{h:>7.2f}" for h in heights))
    out = RESULTS / f"h1_sweep_{mode}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "k2", "final_height_m", "standing"])
        writer.writerows(rows)
    standing = sum(r[3] for r in rows)
    print(f"{standing}/{len(rows)} gain pairs keep H1 standing; wrote {out}")


def view(mode: str, k1: float | None, k2: float | None) -> None:
    import mujoco.viewer

    robot = Humanoid("h1")
    balance = AnkleBalance(robot, mode, k1, k2)
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            start = time.time()
            balance.step()
            viewer.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.time() - start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k1", type=float, help="proportional gain (default depends on the mode)")
    parser.add_argument("--k2", type=float, help="rate gain (default depends on the mode)")
    parser.add_argument("--sweep", choices=["pitch", "com"], help="run a gain grid headless")
    parser.add_argument("--view", choices=list(GAINS), help="open the viewer (run with mjpython)")
    args = parser.parse_args()

    if args.sweep:
        sweep(args.sweep)
    elif args.view:
        view(args.view, args.k1, args.k2)
    else:
        for mode, (k1, k2) in GAINS.items():
            print(f"{mode:>5} (k1={k1}, k2={k2}): base height after 10 s {final_height(mode):.3f} m")


if __name__ == "__main__":
    main()
