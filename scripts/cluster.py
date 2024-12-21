#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import sklearn.cluster as cluster
import visualization
import time
from collections import deque
from geometry_msgs.msg import Pose, PoseStamped
from typing import Type
import os
import json
import io
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class BaseClusterAlgo:
    def __init__(
        self, name: str, algo, args, params, config: visualization.VisualizationConfig
    ):
        self.name = name
        self.labels = None
        self.computation_time = None
        self.algo = algo
        self.algo_args = args if args else []
        self.algo_params = params if params else {}
        self.visualization_config = config

    def calculate_metrics(self):
        raise NotImplementedError()

    def calculate_labels(self, data: np.ndarray):
        try:
            start_time = time.time()
            self.labels = self.algo(*self.algo_args, **self.algo_params).fit_predict(
                data
            )
            end_time = time.time()
            self.computation_time = end_time - start_time
        except TypeError as e:
            print(e)

    def plot(self, data: np.ndarray):
        if self.labels is None:
            raise ValueError("No labels found for plotting")

        plot_title = f"Clusters found by {str(self.algo.__name__)} for {self.name}, took {self.computation_time} seconds"
        return visualization.plot_clusters(
            data, self.labels, plot_title, self.visualization_config
        )


cluster_algos = {}


def register_algo(cls):
    cluster_algos[cls.__name__.lower()] = cls
    return cls


def create_instance(algo_name: str, **kwargs) -> BaseClusterAlgo:
    if algo_name.lower() not in cluster_algos:
        raise ValueError(f"Unknown algorithm {algo_name}, what are you cooking")
    return cluster_algos[algo_name.lower()](**kwargs)


@register_algo
class KMeans(BaseClusterAlgo):
    def __init__(
        self, name: str, args, params, config: visualization.VisualizationConfig
    ):
        super().__init__(name, cluster.KMeans, args, params, config)

    def calculate_metrics(self):
        pass


@register_algo
class DBSCAN(BaseClusterAlgo):
    def __init__(
        self, name: str, args, params, config: visualization.VisualizationConfig
    ):
        super().__init__(name, cluster.DBSCAN, args, params, config)

    def calculate_metrics(self):
        pass


@register_algo
class HDBSCAN(BaseClusterAlgo):
    def __init__(
        self, name: str, args, params, config: visualization.VisualizationConfig
    ):
        super().__init__(name, cluster.HDBSCAN, args, params, config)

    def calculate_metrics(self):
        pass


@register_algo
class AgglomerativeClustering(BaseClusterAlgo):
    def __init__(
        self, name: str, args, params, config: visualization.VisualizationConfig
    ):
        super().__init__(name, cluster.AgglomerativeClustering, args, params, config)

    def calculate_metrics(self):
        pass


class ClusterNode(Node):
    def __init__(self):
        super().__init__("cluster_node")
        self.declare_parameter("name", "default clustering title")
        self.name = self.get_parameter("name").get_parameter_value().string_value
        self.declare_parameter("input_type", "PoseStamped")
        msg_type = {"Pose": Pose, "PoseStamped": PoseStamped}
        self.declare_parameter("input_topic", "")
        self.input_topic_name = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.declare_parameter("output_topic", "")
        self.output_topic_name = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.topic_type = None
        if (
            self.get_parameter("input_type").get_parameter_value().string_value
            in msg_type
        ):
            self.topic_type = msg_type[
                self.get_parameter("input_type").get_parameter_value().string_value
            ]
        else:
            self.get_logger().error("Invalid message type")
            return
        self.declare_parameter("queue_size", 10)
        queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )
        self.queue = deque(maxlen=queue_size)
        self.declare_parameter("algo_params_path", "")
        file_path = (
            self.get_parameter("algo_params_path").get_parameter_value().string_value
        )
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    params = json.load(file)
                algo_name = params["algo_name"]
                if algo_name not in cluster_algos:
                    raise ValueError("check file params and file path")
                algo_params = {
                    key: value for key, value in params.items() if key != "algo_name"
                }
            else:
                self.get_logger().error("File does not exist")
                return
        except KeyError as e:
            self.get_logger().error("Invalid params")
        self.get_logger().info(f"{algo_name} and {algo_params}")
        visualization_config = visualization.VisualizationConfig()
        self.cluster_class = create_instance(
            algo_name=algo_name,
            name=self.name,
            args=(),
            params=algo_params,
            config=visualization_config,
        )
        qos_profile = QoSProfile(depth=10)
        self.msg_sub = self.create_subscription(
            self.topic_type, self.input_topic_name, self.sub_callback, qos_profile
        )
        self.bridge = CvBridge()
        self.visualization_pub = self.create_publisher(
            Image, self.output_topic_name, qos_profile
        )
        self.loop = self.create_timer(3.0, self.timer_callback)

    def sub_callback(self, msg: Type):
        if not isinstance(msg, self.topic_type):
            self.get_logger().error(
                f"Type mismatch, supposed type {self.topic_type}, got {type(msg)}"
            )
            return
        if isinstance(msg, Pose):
            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )
            self.queue.append(position)
        if isinstance(msg, PoseStamped):
            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )
            self.queue.append(position)

    def timer_callback(self):
        if len(self.queue) <= 0:
            self.get_logger().warn("Queue is empty")
        else:
            data = np.array(self.queue)
            self.cluster_class.calculate_labels(data=data)
            frame = self.cluster_class.plot(data=data)
            # Convert to images
            # Convert plot to image
            buf = io.BytesIO()
            frame.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)

            # Convert to OpenCV image
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            # Convert to ROS Image message
            ros2_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros2_image.header.stamp = self.get_clock().now().to_msg()
            ros2_image.header.frame_id = "clustering_frame"
            self.visualization_pub.publish(ros2_image)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
