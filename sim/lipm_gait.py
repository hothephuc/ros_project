"""Step A3: plan a walking CoM trajectory with the linear inverted pendulum model (LIPM).

Kajita's analytic walking pattern: inside each step the ZMP sits on the stance foot and the CoM follows
the closed-form pendulum solution; between steps the next foot placement is corrected so the CoM tracks
the "walk primitive".

    uv run python sim/lipm_gait.py            # print the checkpoint numbers, save results/lipm_gait.png
    uv run python sim/lipm_gait.py --show     # also open the plot window
"""

import argparse
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
G = 9.81
Z_C = 0.70  # CoM height [m]
T_STEP = 0.5  # single-support time per step [s]
STEP_LEN = 0.20  # forward step length [m]
STEP_W = 0.20  # lateral distance between feet [m]
N_STEPS = 8
DT = 0.01
A, B = 10.0, 1.0  # foot-placement weights on position and velocity error


def footsteps(step_len: float = STEP_LEN, step_w: float = STEP_W, n_steps: int = N_STEPS) -> np.ndarray:
    """Foot placements: x advances each step, y alternates left/right; the last step closes the feet."""
    xs = np.arange(n_steps + 1) * step_len
    ys = np.array([(-1) ** (n + 1) * step_w / 2 for n in range(n_steps + 1)])
    xs[-1] = xs[-2]
    return np.stack([xs, ys], axis=1)


def plan(z_c: float = Z_C, t_step: float = T_STEP, feet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the CoM and ZMP trajectories, shape (samples, 2), sampled every DT."""
    feet = footsteps() if feet is None else feet
    tc = np.sqrt(z_c / G)
    c, s = np.cosh(t_step / tc), np.sinh(t_step / tc)

    def walk_primitive(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Ideal CoM state at the END of step n, relative to stance foot n."""
        half = (feet[min(n + 1, len(feet) - 1)] - feet[n]) / 2
        return half, np.array([half[0] * (c + 1) / (tc * s), half[1] * (c - 1) / (tc * s)])

    # Start already in the steady gait: the time-reverse of step 0's end state.
    p = feet[0].copy()
    (x_bar, y_bar), (vx, vy) = walk_primitive(0)
    pos, vel = p + np.array([-x_bar, y_bar]), np.array([vx, -vy])

    com, zmp = [], []
    for n in range(len(feet) - 1):
        for t in np.arange(0, t_step, DT):
            com.append((pos - p) * np.cosh(t / tc) + tc * vel * np.sinh(t / tc) + p)
            zmp.append(p.copy())
        pos, vel = (pos - p) * c + tc * vel * s + p, (pos - p) / tc * s + vel * c
        # Choose the next ZMP p to minimise A*|pos error|^2 + B*|vel error|^2 at the end of step n+1.
        target_p, target_v = walk_primitive(n + 1)
        xd, vd = feet[n + 1] + target_p, target_v
        d = A * (c - 1) ** 2 + B * (s / tc) ** 2
        p = -A * (c - 1) / d * (xd - c * pos - tc * s * vel) - B * s / (tc * d) * (vd - s / tc * pos - c * vel)
    return np.array(com), np.array(zmp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="open the plot window")
    args = parser.parse_args()

    com, zmp = plan()
    print(f"{len(com)} samples, CoM travelled {com[-1, 0]:.2f} m forward")
    print(f"max lateral CoM sway {np.abs(com[:, 1]).max():.3f} m (feet at ±{STEP_W / 2:.2f} m)")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(com)) * DT
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    for i, name in enumerate(["x (forward)", "y (sideways)"]):
        ax[i].plot(t, com[:, i], label="CoM")
        ax[i].step(t, zmp[:, i], where="post", label="ZMP (stance foot)")
        ax[i].set_ylabel(f"{name} [m]")
    ax[0].legend()
    ax[0].set_title("LIPM walking plan: the CoM glides while the ZMP jumps foot to foot")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "lipm_gait.png", dpi=150)
    print(f"saved {RESULTS / 'lipm_gait.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
