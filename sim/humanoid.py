"""Shared helpers: load a Menagerie humanoid and hold it in its keyframe pose with joint PD control."""

from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"
SCENES = {
    "g1": MENAGERIE / "unitree_g1" / "scene.xml",
    "h1": MENAGERIE / "unitree_h1" / "scene.xml",
}
KP, KD = 300.0, 10.0  # PD gains, used only for torque (motor) actuators


class Humanoid:
    """A humanoid whose actuators track the joint angles of keyframe 0."""

    def __init__(self, scene: str | Path) -> None:
        path = SCENES.get(str(scene), Path(scene))
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        # Map each actuator to the qpos and qvel entries of the joint it drives.
        joint_ids = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.is_position_servo = self.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE
        self.lo, self.hi = self.model.actuator_ctrlrange.T
        self.pelvis = self.model.body(1).id  # the floating base is the first body after the world
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[self.qadr].copy()

    def actuator(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def control(self, offsets: np.ndarray | None = None) -> None:
        """Track q_target (+ optional per-actuator offsets): servos get the angle, motors get PD torque."""
        target = self.q_target if offsets is None else self.q_target + offsets
        q, qd = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
        torque = KP * (target - q) - KD * qd
        self.data.ctrl[:] = np.clip(np.where(self.is_position_servo, target, torque), self.lo, self.hi)

    def push(self, force: np.ndarray) -> None:
        """Apply a world-frame force [N] to the pelvis until the next push() call."""
        self.data.xfrc_applied[self.pelvis, :3] = force

    @property
    def height(self) -> float:
        """z of the floating base: the third number in qpos."""
        return float(self.data.qpos[2])

    @property
    def pitch(self) -> float:
        """Torso pitch [rad] from the base quaternion (w, x, y, z); positive leans forward."""
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))

    @property
    def pitch_rate(self) -> float:
        """Angular velocity about the base's y axis [rad/s]."""
        return float(self.data.qvel[4])
