"""Step A4: train a planar biped (Gymnasium Walker2d-v5) to walk with PPO.

    uv run python rl/train_walker.py --steps 1000000 --no-normalize --name ppo_1m_raw
    uv run python rl/train_walker.py --steps 5000000 --name ppo_5m_norm
    uv run python rl/train_walker.py --steps 5000000 --push 100 --name ppo_5m_push   # A5: random pushes

Each run writes runs/<name>/: model.zip, vecnormalize.pkl (if normalized) and progress.csv (the training log).
"""

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from push_wrapper import RandomPush

RUNS = Path(__file__).resolve().parents[1] / "runs"
ENV_ID = "Walker2d-v5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--name", default="ppo_5m_norm")
    parser.add_argument("--no-normalize", action="store_true", help="skip VecNormalize")
    parser.add_argument("--push", type=float, default=0.0, help="max random push on the torso during training [N]")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    wrapper = (lambda env: RandomPush(env, args.push)) if args.push > 0 else None
    env = make_vec_env(ENV_ID, n_envs=8, seed=args.seed, wrapper_class=wrapper)
    if not args.no_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, n_steps=512, batch_size=256, learning_rate=3e-4, seed=args.seed, device="cpu")
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    start = time.time()
    model.learn(total_timesteps=args.steps)
    minutes = (time.time() - start) / 60
    print(f"trained {args.steps:,} steps in {minutes:.1f} min ({args.steps / minutes / 60:,.0f} steps/s)")

    model.save(out / "model")
    if isinstance(env, VecNormalize):
        env.save(str(out / "vecnormalize.pkl"))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
