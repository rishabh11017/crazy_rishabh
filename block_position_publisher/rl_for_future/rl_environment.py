#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

class CrazyflieEnv(Node):
    def __init__(self):
        super().__init__('rl_environment')
        # ====================================
        # RL Action Publisher
        # ====================================
        self.action_pub = self.create_publisher(
            Float64MultiArray,'/rl_action', 10)
        # ====================================
        # Subscribers
        # ====================================
        self.create_subscription(   Odometry,
            '/model/quadcopter/odometry',  self.odom_callback,  10 )
        self.create_subscription( Imu, '/imu/data',
            self.imu_callback,  10   )

        # ====================================
        # State Variables
        # ====================================
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0

        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        # Goal
        # ====================================
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 1.0
        self.goal_phi = 0.0
        # Command States
        # ====================================
        self.Fx = 0.0
        self.psi_d = 0.0
        self.pz_d = 1.0
        # ====================================
        self.goal_threshold = 0.20
        self.max_distance = 5.0
        self.min_height = 0.05
        self.max_roll = np.deg2rad(80)
        self.max_pitch = np.deg2rad(80)
                # Episode Variables
        # ====================================
        self.step_count = 0
        self.max_steps = 100
        self.prev_distance = 0.0
        self.prev_action = np.zeros(3)
        self.get_logger().info(
            "RL Environment Ready"
        )
    # =====================================================
    # IMU CALLBACK
    # =====================================================

    def imu_callback(self,msg):

        self.ax = msg.linear_acceleration.x
        self.ay = msg.linear_acceleration.y
        self.az = msg.linear_acceleration.z

        self.wx = msg.angular_velocity.x
        self.wy = msg.angular_velocity.y
        self.wz = msg.angular_velocity.z

    # =====================================================
    # ODOM CALLBACK
    # =====================================================

    def odom_callback(self,msg):
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        self.pz = msg.pose.pose.position.z

        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        rot = R.from_quat([qx,qy,qz,qw] )
        self.phi,self.theta,self.psi = rot.as_euler( 'xyz', degrees=False)
    # GOAL GENERATION take random goal position within a certain range
    # =====================================================
    def generate_goal(self):
        self.goal_x = np.random.uniform( -2.0,2.0 )
        self.goal_y = np.random.uniform( -2.0, 2.0 )
        self.goal_z = np.random.uniform(0.5, 2.0)
    # DISTANCE TO GOAL distance between current position and goal position
    # =====================================================
    def distance_to_goal(self):
        return np.linalg.norm(
            [ self.px-self.goal_x,
                self.py-self.goal_y,
                self.pz-self.goal_z ] )
    # OBSERVATION
    # =====================================================
    def get_observation(self):
        obs = np.array([
            self.ax, self.ay, self.az,
            self.wx, self.wy, self.wz,
            self.phi, self.theta,self.psi,
            self.pz, self.vz,
            self.goal_x, self.goal_y,self.goal_z   ],dtype=np.float32)
        return obs
    # RESET
    # =====================================================
    def reset(self):
        self.generate_goal()
        self.step_count = 0
        self.prev_action = np.zeros(3)
        self.prev_distance = self.distance_to_goal()
        obs = self.get_observation()
        return obs
    # APPLY ACTION
    # =====================================================
    def apply_action(self,action):
        dFx = float(action[0])
        dPsi = float(action[1])
        dPz = float(action[2])
        self.Fx += dFx
        self.psi_d += dPsi
        self.pz_d += dPz
        msg = Float64MultiArray()
        msg.data = [  self.pz_d,   0.0, 0.0,  self.psi_d,
            self.Fx ]     # as my new controller takes 5d vector input 
        self.action_pub.publish(msg)
    # REWARD
    # =====================================================
    def compute_reward(self,action):
        d = self.distance_to_goal()
        progress = self.prev_distance - d
        reward = 10.0*progress
        reward -= 0.01
        reward -= 0.02*(self.phi**2 +self.theta**2)
        reward -= 0.02*np.linalg.norm(  action-self.prev_action )
        goal_vec = np.array([
            self.goal_x-self.px, self.goal_y-self.py,
            self.goal_z-self.pz   ])
        vel = np.array([self.vx, self.vy, self.vz])
        reward += 0.1*(np.dot(goal_vec,vel) /
    (np.linalg.norm(goal_vec)+1e-6))
        if d < 0.2:
            reward += 400.0    
        if self.pz < 0.05:
            reward -= 100.0
        self.prev_distance = d
        self.prev_action = action.copy()
        return reward
    # DONE
    # =====================================================
    def check_done(self):
        d=self.distance_to_goal()
        if self.distance_to_goal() < self.goal_threshold:
            return True
        if self.pz < self.min_height:
            return True
        if self.step_count > self.max_steps:
            return True
        if d>=self.max_distance:
            return True
        if abs(self.phi) > self.max_roll:
            return True
        if abs(self.theta) > self.max_pitch:
            return True
        return False
    # STEP
    # =====================================================
    def step(self,action):
        self.step_count += 1
        self.apply_action(action)
        obs = self.get_observation()
        reward = self.compute_reward(action)
        done = self.check_done()
        info = {}
        return obs,reward,done,info
def main():
    rclpy.init()
    env = CrazyflieEnv()
    env.reset()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()