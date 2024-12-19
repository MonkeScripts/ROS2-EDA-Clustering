#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import sklearn.cluster as cluster
import visualization
import time
from collections import defaultdict, deque
from geometry_msgs.msg import Pose, PoseStamped
from typing import Type

class BaseClusterAlgo:
    def __init__(self, name:str, data:np.ndarray, algo, args, params, config:visualization.VisualizationConfig):
        self.name = name
        self.data = data
        self.labels = None
        self.computation_time = None
        self.algo = algo
        self.algo_args = args if args else []
        self.algo_params = params if params else {}
        self.visualization_config = config

    def calculate_metrics(self):
        raise NotImplementedError()

    def calculate_labels(self):
        try:
            start_time = time.time()
            self.labels = self.algo(*self.algo_args, **self.algo_params).fit_predict(
                self.data
            )
            end_time = time.time()
            self.computation_time = end_time - start_time
        except TypeError as e:
            print(e)

    def plot(self):
        if self.labels is None:
            raise ValueError("No labels found for plotting")

        plot_title = f"Clusters found by {str(self.algo.__name__)} for {self.name}, took {self.computation_time} seconds"
        return visualization.plot_clusters(self.data, self.labels, plot_title, self.visualization_config)

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
    def __init__(self, name:str, data:np.ndarray, args, params, config: visualization.VisualizationConfig):
        super().__init__(name, data, cluster.KMeans, args, params, config)

    def calculate_metrics(self):
        pass

@register_algo
class DBSCAN(BaseClusterAlgo):
    def __init__(self, name:str, data:np.ndarray, args, params, config: visualization.VisualizationConfig):
        super().__init__(name, data, cluster.DBSCAN, args, params, config)

    def calculate_metrics(self):
        pass

@register_algo
class HDBSCAN(BaseClusterAlgo):
    def __init__(self, name:str, data:np.ndarray, args, params, config: visualization.VisualizationConfig):
        super().__init__(name, data, cluster.HDBSCAN, args, params, config)
    def calculate_metrics(self):
        pass
@register_algo
class AgglomerativeClustering(BaseClusterAlgo):
    def __init__(self, name:str, data:np.ndarray, args, params, config: visualization.VisualizationConfig):
        super().__init__(name, data, cluster.AgglomerativeClustering, args, params, config)

    def calculate_metrics(self):
        pass

class ClusterNode(Node): 
    def __init__(self):
        super().__init__('cluster_node') 
        self.declare_parameter("input_type", "PoseStamped")
        msg_type = {
            "Pose" : Pose,
            "PoseStamped" : PoseStamped
        }
        self.declare_parameter("input_topic", "")
        self.input_topic_name = self.get_parameter("input_topic").get_parameter_value().string_value
        self.declare_parameter("output_topic", "")
        self.output_topic_name = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.topic_type = None
        if self.get_parameter("input_type").get_parameter_value().string_value in msg_type: 
            self.topic_type = msg_type[self.get_parameter("input_type").get_parameter_value().string_value]
        else:
            self.get_logger().error("Invalid message type")
            return
        self.declare_parameter("queue_size", 10)
        queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self.queue = deque(maxlen=queue_size)
        self.declare_parameter("algo_name", "kmeans")
        #TODO use JSON file instead to load specific algo params
        # self.algo_name = self.get_parameter("algo_name").get_parameter_value().string_value.lower()
        # if self.algo_name.lower() not in cluster_algos:
        #     self.get_logger().error("Invalid clustering algo")
        #     return
        # self.declare_parameter("algo_params", value={})
        self.algo_params = self.get_parameter("algo_params").get_parameter_value()
        self.get_logger().info(f"{self.algo_params}....")
        self.cluster_class = create_instance(self.algo_name, self.algo_params)
        qos_profile = QoSProfile(depth=10)
        self.msg_sub = self.create_subscription(self.topic_type, self.input_topic_name, self.sub_callback(), qos_profile)
        self.clustered_pub = self.create_publisher(self.topic_type, self.output_topic_name, qos_profile)
        self.loop = self.create_timer(3.0, self.timer_callback)

    def sub_callback(self, msg: Type):
        if not isinstance(msg, self.topic_type):
            self.get_logger().error(f"Type mismatch, supposed type {self.topic_type}, got {type(msg)}")
            return
        if isinstance(msg, Pose):
            position = np.array([msg.position.x, msg.position.y, msg.position.z])
            self.queue.append(position)
        if isinstance(msg, PoseStamped):
            position = np.array([msg.position.x, msg.position.y, msg.position.z])
            self.queue.append(position)

    def timer_callback(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
