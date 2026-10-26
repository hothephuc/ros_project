"""Step A4/A5: evaluate a trained Walker2d policy deterministically, optionally under pushes, or watch it.

    uv run python rl/evaluate.py ppo_5m_norm                     # 5 episodes: return and length
    uv run python rl/evaluate.py ppo_5m_norm --push 150          # same, with random 150 N pushes
    uv run python rl/evaluate.py ppo_5m_norm --watch             # open a window (try mjpython if it won't)
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from push_wrapper import RandomPush

RUNS = Path(__file__).resolve().parents[1] / "runs"
ENV_ID = "Walker2d-v5"


def load(name: str, push: float = 0.0, seed: int = 1000, render: bool = False) -> tuple[PPO, VecEnv]:
    run = RUNS / name
    wrapper = (lambda env: RandomPush(env, push, fixed=True)) if push > 0 else None
    env_kwargs = {"render_mode": "human"} if render else None
    env = make_vec_env(ENV_ID, n_envs=1, seed=seed, wrapper_class=wrapper, env_kwargs=env_kwargs)
    if (run / "vecnormalize.pkl").exists():
        env = VecNormalize.load(str(run / "vecnormalize.pkl"), env)
        env.training, env.norm_reward = False, False  # freeze the statistics, report the true reward
    return PPO.load(run / "model", device="cpu"), env


def evaluate(name: str, episodes: int = 5, push: float = 0.0, seed: int = 1000) -> list[tuple[float, int]]:
    """Return (return, length) for each deterministic episode."""
    model, env = load(name, push, seed)
    results = []
    obs = env.reset()
    while len(results) < episodes:
        total, length, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = env.step(action)  # the VecEnv resets itself when an episode ends
            total, length, done = total + float(reward[0]), length + 1, bool(dones[0])
        results.append((total, length))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="run folder under runs/")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--push", type=float, default=0.0, help="push magnitude during evaluation [N]")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        model, env = load(args.name, args.push, render=True)
        obs = env.reset()
        for _ in range(3000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
        return

    results = evaluate(args.name, args.episodes, args.push)
    returns = np.array([r for r, _ in results])
    print(f"{args.name}, push {args.push:.0f} N")
    print("returns:", ", ".join(f"{r:,.0f}" for r in returns))
    print("lengths:", ", ".join(str(n) for _, n in results))
    print(f"mean {returns.mean():,.0f} ± {returns.std():,.0f}")


if __name__ == "__main__":
    main()
