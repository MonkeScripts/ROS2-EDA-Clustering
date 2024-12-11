#!/usr/bin/env python3
import rclpy
from rclpy.node import Node



class ClusterNode(Node): #MODIFY NAME
    def __init__(self):
        super().__init__('cluster_node') #MODIFY NAME
        self.get_logger().info("hello")


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()