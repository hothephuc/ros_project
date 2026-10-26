"""Step 7: record the README demo GIF offscreen (no screen recording needed).

Four clips: G1 taking a push, H1 with no balance feedback, H1 with CoM feedback taking a push,
and the trained Walker2d policy walking. Needs runs/ppo_5m_norm (see rl/train_walker.py).

    uv run python scripts/record_gif.py     # writes results/demo.gif
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "rl")]

from balance_h1 import AnkleBalance  # noqa: E402
from humanoid import Humanoid  # noqa: E402

RESULTS = ROOT / "results"
WIDTH, HEIGHT, FPS = 400, 270, 12
SECONDS = 4.0


def caption(frame: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 26], fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return image


def humanoid_clip(scene: str, text: str, mode: str | None = None, push: float = 0.0) -> list[Image.Image]:
    """Simulate a Menagerie humanoid, optionally with balance feedback and a push at t = 1 s."""
    robot = Humanoid(scene)
    balance = AnkleBalance(robot, mode) if mode else None
    renderer = mujoco.Renderer(robot.model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = robot.pelvis
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -15.0

    frames, next_frame = [], 0.0
    while robot.data.time < SECONDS:
        pushing = 1.0 <= robot.data.time < 1.1
        robot.push(np.array([push if pushing else 0.0, 0.0, 0.0]))
        if balance:
            balance.step()
        else:
            robot.control()
            mujoco.mj_step(robot.model, robot.data)
        if robot.data.time >= next_frame:
            renderer.update_scene(robot.data, camera)
            label = f"{text}   (PUSH {push:.0f} N)" if pushing and push else text
            frames.append(caption(renderer.render(), label))
            next_frame += 1.0 / FPS
    renderer.close()
    return frames


def walker_clip(run: str = "ppo_5m_norm") -> list[Image.Image]:
    """Run the trained PPO policy and grab rgb frames."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env("Walker2d-v5", n_envs=1, seed=1000)
    env = VecNormalize.load(str(ROOT / "runs" / run / "vecnormalize.pkl"), env)
    env.training, env.norm_reward = False, False
    model = PPO.load(ROOT / "runs" / run / "model", device="cpu")

    walker = env.venv.envs[0].unwrapped
    renderer = mujoco.Renderer(walker.model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()  # side view that follows the torso
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = walker.model.body("torso").id
    camera.distance, camera.azimuth, camera.elevation = 4.0, 90.0, -10.0

    frames = []
    obs = env.reset()
    step_dt = walker.dt  # 8 ms per env step
    every = max(1, round(1.0 / FPS / step_dt))
    for i in range(int(SECONDS * 1.5 / step_dt)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        if i % every == 0:
            renderer.update_scene(walker.data, camera)
            frames.append(caption(renderer.render(), "Walker2d: PPO policy learned in 8 min (5M steps)"))
    renderer.close()
    env.close()
    return frames


def main() -> None:
    frames = []
    frames += humanoid_clip("g1", "A1: Unitree G1 holds its pose with joint servos", push=75)
    frames += humanoid_clip("h1", "A2: Unitree H1, joint PD only -> falls")
    frames += humanoid_clip("h1", "A2/A5: H1 + CoM ankle feedback survives a push", mode="com", push=150)
    frames += walker_clip()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"saved {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
