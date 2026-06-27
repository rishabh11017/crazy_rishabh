#!/usr/bin/env python3
"""
WSL2 ROS2 Controller Receiver
Receives controller data via UDP and publishes setpoints to /desired_trajectory
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import socket
import json
import threading
import time
import numpy as np

class ControllerReceiver(Node):
    def __init__(self):
        super().__init__('controller_receiver')
        
        self.publisher = self.create_publisher(Float64MultiArray, '/desired_trajectory', 10)
        
        self.pd = 5.0
        self.vd = 0.0
        self.ad = 0.0
        self.psi_d = 0.0
        self.F_x = 0.0
        
        self.fx_scale = -0.15
        self.prev_buttons = None
        
        # Store the state of each button: 0=released, 1=just_pressed, 2=held
        self.button_state = {}
        
        # Debounce delay
        self.debounce_delay = 0.5
        
        self.udp_port = 5555
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port))
        self.sock.settimeout(0.1)
        
        self.get_logger().info('Controller Receiver Node Started')
        self.get_logger().info('  Y button -> pd +0.1  |  A button -> pd -0.1')
        self.get_logger().info('  X button -> psi_d +5 deg  |  B button -> psi_d -5 deg')
        self.get_logger().info('  Left Stick Y -> F_x')
        
        self.running = True
        self.thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.thread.start()
    
    def publish_setpoints(self):
        msg = Float64MultiArray()
        msg.data = [self.pd, self.vd, self.ad, self.psi_d, self.F_x]
        self.publisher.publish(msg)
        self.get_logger().info(f'Setpoints: pd={self.pd:.2f}, psi_d={self.psi_d:.2f}, F_x={self.F_x:.2f}')
    
    def is_new_press(self, button_id, current_pressed):
        """Returns True only on the very first press, ignores until release"""
        if button_id not in self.button_state:
            self.button_state[button_id] = False
        
        if current_pressed and not self.button_state[button_id]:
            # Just pressed - this is the rising edge
            self.button_state[button_id] = True
            return True
        elif not current_pressed:
            # Button released - reset state
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
                
                # Y button (index 3) -> pd +0.1
                if self.is_new_press(3, buttons[3] == 1):
                    self.pd += 0.1
                    changed = True
                    self.get_logger().info(f'Y pressed: pd = {self.pd:.2f}')
                
                # A button (index 0) -> pd -0.1
                if self.is_new_press(0, buttons[0] == 1):
                    self.pd -= 0.1
                    changed = True
                    self.get_logger().info(f'A pressed: pd = {self.pd:.2f}')
                
                # X button (index 2) -> psi_d +5
                if self.is_new_press(2, buttons[2] == 1):
                    self.psi_d += np.radians(30)  # 30 degrees in radians
                    changed = True
                    self.get_logger().info(f'X pressed: psi_d = {np.degrees(self.psi_d):.2f} deg')
                
                # B button (index 1) -> psi_d -5
                if self.is_new_press(1, buttons[1] == 1):
                    self.psi_d -= np.radians(30)  # 30 degrees in radians
                    changed = True
                    self.get_logger().info(f'B pressed: psi_d = {np.degrees(self.psi_d):.2f} deg')
                
                # Left joystick Y (axis 1) -> F_x
                new_F_x = axes[3] * self.fx_scale
                if abs(new_F_x) < 0.05:
                    new_F_x = 0.0
                
                if abs(new_F_x - self.F_x) > 0.01:
                    self.F_x = round(new_F_x, 3)
                    changed = True
                
                if changed:
                    self.publish_setpoints()
                
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