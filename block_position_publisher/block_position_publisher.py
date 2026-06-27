#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_pb2 import Pose

class BlockPositionPublisher(Node):
    def __init__(self):
        super().__init__('block_position_publisher')
        
        # Create publisher for block position
        self.publisher_ = self.create_publisher(
            PoseStamped,
            'block/pose',
            10
        )
        
        # Create Gazebo Transport node
        self.gz_node = GzNode()
        
        # Subscribe to pose topic from Gazebo
        # The topic format is: /model/<model_name>/pose
        self.gz_node.subscribe(
            '/model/block/pose',
            self.pose_callback,
            Pose
        )
        
        self.get_logger().info('Block Position Publisher started')
        self.get_logger().info('Waiting for block pose messages from Gazebo...')
        
    def pose_callback(self, msg):
        """Callback for Gazebo pose messages"""
        try:
            # Create ROS PoseStamped message
            ros_msg = PoseStamped()
            
            # Set header
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = 'world'
            
            # Copy position
            ros_msg.pose.position.x = msg.position.x
            ros_msg.pose.position.y = msg.position.y
            ros_msg.pose.position.z = msg.position.z
            
            # Copy orientation
            ros_msg.pose.orientation.x = msg.orientation.x
            ros_msg.pose.orientation.y = msg.orientation.y
            ros_msg.pose.orientation.z = msg.orientation.z
            ros_msg.pose.orientation.w = msg.orientation.w
            
            # Publish
            self.publisher_.publish(ros_msg)
            
            # Log position (reduced frequency to avoid spam)
            self.get_logger().debug(
                f'Block position: x={msg.position.x:.3f}, '
                f'y={msg.position.y:.3f}, z={msg.position.z:.3f}'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {str(e)}')
    
    def spin(self):
        """Spin function to process Gazebo Transport messages"""
        while rclpy.ok():
            # Process Gazebo Transport messages
            self.gz_node.spin_once(timeout_ms=10)
            
            # Spin ROS once
            rclpy.spin_once(self, timeout_sec=0.01)

def main(args=None):
    rclpy.init(args=args)
    
    node = BlockPositionPublisher()
    
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()