#!/usr/bin/env python3

import socket
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


class UDPListener(Node):

    def __init__(self):
        super().__init__("udp_listener")

        # ROS2 publisher
        self.publisher = self.create_publisher(
            Vector3,
            "udp_data",
            10
        )

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5005))   # Listen on port 5005
        self.sock.setblocking(False)

        # Check for UDP packets every 10 ms
        self.timer = self.create_timer(0.001, self.receive_udp)

        self.get_logger().info("Listening on UDP port 5005")

    def receive_udp(self):
        try:
            data, addr = self.sock.recvfrom(1024)

            # Decode JSON packet
            packet = json.loads(data.decode())

            x = float(packet["x"])
            y = float(packet["y"])
            z = float(packet["z"])

            # Publish as Vector3
            msg = Vector3()
            msg.x = x
            msg.y = y
            msg.z = z

            self.publisher.publish(msg)

            self.get_logger().info(
                f"Received from {addr}: x={x:.4f}, y={y:.4f}, z={z:.4f}"
            )

        except BlockingIOError:
            # No packet available
            pass

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().error(f"Invalid packet: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = UDPListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()