"""Step B4: make the biped squat by sending trajectories to the leg controller.

The knee bends by 2d while hip pitch and ankle pitch each move by -d. Because the thigh and shin
are the same length, the hip stays straight above the ankle and the foot stays flat.

    ros2 run biped_control squat --ros-args -p depth:=0.4 -p period:=4.0
"""

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [
    "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch",
]


def leg_pose(d: float) -> list[float]:
    """Joint angles for a squat of depth d [rad]: hip -d, knee 2d, ankle -d, on both legs."""
    one_leg = [0.0, -d, 2.0 * d, -d]
    return one_leg + one_leg


def seconds(t: float) -> Duration:
    return Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))


class Squat(Node):
    def __init__(self) -> None:
        super().__init__("squat")
        self.depth = self.declare_parameter("depth", 0.4).value
        self.period = self.declare_parameter("period", 4.0).value
        self.publisher = self.create_publisher(JointTrajectory, "/leg_controller/joint_trajectory", 10)
        self.timer = self.create_timer(self.period, self.send)
        self.get_logger().info(f"squatting to depth {self.depth} rad every {self.period} s")

    def send(self) -> None:
        """One squat: stand -> down (first half of the period) -> back up (second half)."""
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        for t, d in [(self.period * 0.5, self.depth), (self.period * 0.95, 0.0)]:
            msg.points.append(JointTrajectoryPoint(positions=leg_pose(d), time_from_start=seconds(t)))
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = Squat()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
