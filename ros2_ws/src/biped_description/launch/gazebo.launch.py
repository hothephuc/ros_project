"""Step B3: start Gazebo, spawn the biped, bridge /clock and start both controllers.

    ros2 launch biped_description gazebo.launch.py              # free-standing, spawned 0.85 m up
    ros2 launch biped_description gazebo.launch.py fixed:=true  # torso bolted to the world
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("biped_description")
    xacro_file = os.path.join(share, "urdf", "biped.urdf.xacro")
    controllers = os.path.join(share, "config", "controllers.yaml")
    fixed = LaunchConfiguration("fixed")

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " sim:=true", " fixed:=", fixed, " controllers:=", controllers]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-r empty.sdf"}.items(),
    )
    state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )
    spawn = Node(
        package="ros_gz_sim", executable="create",
        arguments=["-topic", "robot_description", "-name", "biped", "-z", "0.85"],
    )
    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )
    joint_state_broadcaster = Node(package="controller_manager", executable="spawner",
                                   arguments=["joint_state_broadcaster"])
    leg_controller = Node(package="controller_manager", executable="spawner", arguments=["leg_controller"])

    return LaunchDescription([
        DeclareLaunchArgument("fixed", default_value="false", description="bolt the torso to the world"),
        gazebo,
        state_publisher,
        spawn,
        clock_bridge,
        # Start the controllers only once the robot (and its controller manager) exists in Gazebo.
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(target_action=joint_state_broadcaster, on_exit=[leg_controller])),
    ])
