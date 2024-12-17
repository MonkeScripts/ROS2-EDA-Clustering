#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import sklearn.cluster as cluster
import visualization
import time

cluster_algos = {}

def register_algo(cls):
    cluster_algos[cls.__name__] = cls
    return cls

def create_instance(algo_name):
    if algo_name not in cluster_algos:
        raise ValueError(f"Unknown algorithm {algo_name}, what are you cooking")
    return cluster_algos[algo_name]
import sklearn.cluster as cluster

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
            return None
        plot_title = f"Clusters found by {str(self.algo.__name__)}, took {self.computation_time} seconds"
        return visualization.plot_clusters(self.data, self.labels, plot_title, self.visualization_config)



@register_algo
class KMeans(BaseClusterAlgo):
    def __init__(self, name:str, data:np.ndarray, args, params, config: visualization.VisualizationConfig):
        super().__init__(name, data, cluster.KMeans, args, params, config)

    def calculate_metrics(self):
        return super().calculate_metrics()

class ClusterNode(Node): 
    def __init__(self):
        super().__init__('cluster_node') 
        self.get_logger().info("hello")


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
