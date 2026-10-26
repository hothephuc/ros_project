"""Step A1: hold a humanoid in its standing pose, with or without the viewer, and push-test it.

    uv run python sim/stand.py g1 --headless             # simulate 10 s, print base height
    uv run python sim/stand.py g1 --headless --push 50   # 50 N forward push on the pelvis at t = 2 s
    uv run mjpython sim/stand.py g1                      # interactive viewer (macOS needs mjpython)
"""

import argparse
import time

import mujoco
import numpy as np

from humanoid import Humanoid

PUSH_AT, PUSH_FOR = 2.0, 0.1  # push start time and duration [s]


def simulate(robot: Humanoid, seconds: float = 10.0, push: float = 0.0) -> float:
    """Run headless with an optional forward push; return the final base height."""
    while robot.data.time < seconds:
        pushing = PUSH_AT <= robot.data.time < PUSH_AT + PUSH_FOR
        robot.push(np.array([push if pushing else 0.0, 0.0, 0.0]))
        robot.control()
        mujoco.mj_step(robot.model, robot.data)
    return robot.height


def view(robot: Humanoid) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            start = time.time()
            robot.control()
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.time() - start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", default="g1", help="g1, h1, or a path to a scene.xml")
    parser.add_argument("--headless", action="store_true", help="simulate 10 s without a window")
    parser.add_argument("--push", type=float, default=0.0, help="forward push on the pelvis [N] (headless only)")
    args = parser.parse_args()

    robot = Humanoid(args.scene)
    if args.headless:
        height = simulate(robot, push=args.push)
        print(f"t={robot.data.time:.1f}s  push={args.push:.0f} N  base height={height:.3f} m")
    else:
        view(robot)


if __name__ == "__main__":
    main()
