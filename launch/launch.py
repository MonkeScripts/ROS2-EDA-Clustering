import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import (
    ComposableNodeContainer,
    Node,
    PushRosNamespace,
    SetParameter,
)
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true",
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "namespace",
            default_value="auv",
            description="namespace of vehicle",
        )
    )

    for param in [
        "use_sim_time",
        "namespace",
    ]:
        ld.add_action(LogInfo(msg=[param, ": ", LaunchConfiguration(param)]))
    test_node = Node(
        package="eda_cluster",
        executable="test_node.py",
        name="test_node",
        output="screen",
        parameters=[
            {
            }
        ],
    )

    actions = [
        SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
        PushRosNamespace(LaunchConfiguration("namespace")),
        test_node
    ]

    ld.add_action(GroupAction(actions=actions))

    return ld
