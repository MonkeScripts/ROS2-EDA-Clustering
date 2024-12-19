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
    cluster_node = Node(
        package="eda_cluster",
        executable="cluster.py",
        name="cluster_node",
        output="screen",
        parameters=[
            {
                "input_type": "PoseStamped",
                "input_topic": "/in",
                "output_topic": "/out",
                "queue_size": 10,
                "algo_name": "kmeans",
                "algo_params": {"n_clusters": 3},
            }
        ],
    )

    actions = [
        SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
        PushRosNamespace(LaunchConfiguration("namespace")),
        cluster_node
    ]

    ld.add_action(GroupAction(actions=actions))

    return ld
