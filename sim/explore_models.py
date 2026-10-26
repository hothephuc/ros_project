"""Step 1: load Menagerie humanoids and print model dimensions."""

from pathlib import Path

import mujoco

MENAGERIE = Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie"


def describe(label: str, scene_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    print(f"=== {label} ({scene_path.name}) ===")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nkey={model.nkey}")
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        print(f"  keyframe: {name}")
    print("actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i:2d} {name}")
    print()


def main() -> None:
    describe("Unitree G1", MENAGERIE / "unitree_g1" / "scene.xml")
    describe("Unitree H1", MENAGERIE / "unitree_h1" / "scene.xml")


if __name__ == "__main__":
    main()
