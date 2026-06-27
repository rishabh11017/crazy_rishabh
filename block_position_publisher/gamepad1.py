#!/usr/bin/env python3
"""
WSL2 ROS2 Controller Receiver
Receives controller data via UDP and publishes setpoints to /quadcopter/setpoint
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import socket
import json
import threading
from std_msgs.msg import Float64MultiArray


class ControllerReceiver(Node):
    def __init__(self):
        super().__init__('controller_receiver')
        
        self.publisher = self.create_publisher(Float64MultiArray, '/desired_trajectory', 10)
        
        # Accumulated setpoints
        self.pd_x = 0.0
        self.pd_y = 0.0
        self.pd_z = 2.0  # start at hover altitude
        self.psi_d = 0.0
        
        # Button state tracking for edge detection
        self.button_state = {}
        
        # Right stick tracking for edge detection
        self.right_stick_prev = 0.0
        self.right_stick_threshold = 0.5
        
        # UDP setup
        self.udp_port = 5555
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.settimeout(0.1)
        
        self.get_logger().info('Controller Receiver Node Started')
        self.get_logger().info('Mapping:')
        self.get_logger().info('  A -> pd.y +0.1  |  B -> pd.x +0.1')
        self.get_logger().info('  X -> pd.y -0.1  |  Y -> pd.x -0.1')
        self.get_logger().info('  Right Stick Up   -> pd.z +0.1')
        self.get_logger().info('  Right Stick Down -> pd.z -0.1')
        
        self.running = True
        self.thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.thread.start()
    
    def publish_setpoint(self):
        msg = Float64MultiArray()
        msg.data = [self.pd_x, self.pd_y, self.pd_z, self.psi_d]
        self.publisher.publish(msg)
        self.get_logger().info(
            f'Setpoint: x={self.pd_x:.1f}, y={self.pd_y:.1f}, z={self.pd_z:.1f}'
        )
    
    def is_new_press(self, button_id, current_pressed):
        """Rising edge detection - fires once per press"""
        if button_id not in self.button_state:
            self.button_state[button_id] = False
        
        if current_pressed and not self.button_state[button_id]:
            self.button_state[button_id] = True
            return True
        elif not current_pressed:
            self.button_state[button_id] = False
        
        return False
    
    def receive_loop(self):
        while self.running and rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(1024)
                msg_data = json.loads(data.decode())
                buttons = msg_data["buttons"]
                axes = msg_data["axes"]
                
                changed = False
                
                # A button (index 0) -> pd.y +0.1
                if self.is_new_press(0, buttons[3] == 1):
                    self.pd_x += 1
                    changed = True
                    self.get_logger().info(f'A pressed: pd.y = {self.pd_y:.1f}')
                
                # B button (index 1) -> pd.x +0.1
                if self.is_new_press(1, buttons[2] == 1):
                    self.pd_y -= 1
                    changed = True
                    self.get_logger().info(f'B pressed: pd.x = {self.pd_x:.1f}')
                
                # X button (index 2) -> pd.y -0.1
                if self.is_new_press(2, buttons[0] == 1):
                    self.pd_y += 1
                    changed = True
                    self.get_logger().info(f'X pressed: pd.y = {self.pd_y:.1f}')
                
                # Y button (index 3) -> pd.x -0.1
                if self.is_new_press(3, buttons[1] == 1):
                    self.pd_x -= 1
                    changed = True
                    self.get_logger().info(f'Y pressed: pd.x = {self.pd_x:.1f}')
                if self.is_new_press(4, buttons[4] == 1):  # LB button
                    self.psi_d += 0.1
                    changed = True
                    self.get_logger().info(f'LB pressed: psi_d = {self.psi_d:.2f}')
                if self.is_new_press(5, buttons[5] == 1):  # RB button
                    self.psi_d -= 0.1
                    changed = True
                    self.get_logger().info(f'RB pressed: psi_d = {self.psi_d:.2f}')
                
                # Right stick Y (axis 3) -> pd.z control with edge detection
                right_stick = axes[3]
                
                # Up: stick goes negative in many controllers
                if right_stick < -self.right_stick_threshold and self.right_stick_prev >= -self.right_stick_threshold:
                    self.pd_z += 1
                    changed = True
                    self.get_logger().info(f'Right Stick Up: pd.z = {self.pd_z:.1f}')
                
                # Down: stick goes positive
                if right_stick > self.right_stick_threshold and self.right_stick_prev <= self.right_stick_threshold:
                    self.pd_z -= 1
                    changed = True
                    self.get_logger().info(f'Right Stick Down: pd.z = {self.pd_z:.1f}')
                
                self.right_stick_prev = right_stick
                
                if changed:
                    self.publish_setpoint()
                
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Error: {e}')
    
    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()