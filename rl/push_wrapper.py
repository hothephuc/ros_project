"""A Gymnasium wrapper that shoves the Walker2d torso sideways (along x) at random times."""

import gymnasium as gym
import numpy as np

PUSH_STEPS = 5  # each push lasts 5 env steps (5 x 8 ms = 40 ms)


class RandomPush(gym.Wrapper):
    """With probability `rate` per step, start a push of random sign and magnitude up to `max_force` [N]."""

    def __init__(self, env: gym.Env, max_force: float, rate: float = 0.01, fixed: bool = False) -> None:
        super().__init__(env)
        self.max_force, self.rate, self.fixed = max_force, rate, fixed
        self.torso = env.unwrapped.model.body("torso").id
        self.remaining = 0

    def reset(self, **kwargs):
        self.remaining = 0
        self._set_force(0.0)
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.remaining == 0 and self.np_random.random() < self.rate:
            magnitude = self.max_force if self.fixed else self.np_random.uniform(0.0, self.max_force)
            self._set_force(self.np_random.choice([-1.0, 1.0]) * magnitude)
            self.remaining = PUSH_STEPS
        elif self.remaining > 0:
            self.remaining -= 1
            if self.remaining == 0:
                self._set_force(0.0)
        return self.env.step(action)

    def _set_force(self, fx: float) -> None:
        self.env.unwrapped.data.xfrc_applied[self.torso, :3] = np.array([fx, 0.0, 0.0])
