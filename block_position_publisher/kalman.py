import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, Wrench
from nav_msgs.msg import Odometry
import pandas as pd
from sensor_msgs.msg import Imu


class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter')
         # ── Physical parameters ────────────────────────────────────
        self.m = 0.031
        self.g = 9.81
        self.J = np.diag([6.4e-4, 6.4e-4, 1.28e-3])
        self.get_logger().info(f"Controller mass = {self.m}")

         # ── Current state (updated by subscribers) ─────────────────
        self.p     = np.array([0.0, 0.0, 0])                       # position
        self.q     = np.array([1.0, 0.0, 0.0, 0.0])   # quaternion [w, x, y, z]
        self.v     = np.zeros(3)                        # linear velocity
        self.omega = np.zeros(3)                        # angular vel

          # ── susbscriptions ─────────────────────────────────────────────
        self.create_subscription(  Odometry,'/model/quadcopter/odometry', self.odom_callback,  1 )
        self.command_sub = self.create_subscription(  Float64MultiArray,  '/desired_trajectory', self.command_callback, 10)
        #this is for receiving the desired trajectory from gamepad receiver node
        self.force_pub = self.create_publisher(Wrench, '/block/force', 1)
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)

        def imu_callback(self, msg: Imu):
            """Extract orientation, angular velocity, and linear acceleration from /imu/data."""
            # 1. Save orientation quaternion matching his [w, x, y, z] order style
            self.q = np.array([
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ])
        
    # 2. Save angular velocity (omega) from the IMU
            self.omega = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])
        
            # 3. Save linear acceleration from the IMU
            self.accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ])
        def odom_callback(self, msg: Odometry):
            """Extract linear and angular velocity from /model/quadcopter/odometry."""
            self.p_true = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])
        
            self.q_true = np.array([
                 msg.pose.pose.orientation.w,
                 msg.pose.pose.orientation.x,
                 msg.pose.pose.orientation.y,
                 msg.pose.pose.orientation.z,
             ])
            
            self.v_true = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ])
        
            self.omega_true = np.array([
                 msg.twist.twist.angular.x,
                 msg.twist.twist.angular.y,
                 msg.twist.twist.angular.z,
             ])
        
        def quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
            """Convert quaternion to rotation matrix."""
            w, x, y, z = q
            R = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
            ])
            return R
        def yaw_pitch_roll_from_quaternion(self, q: np.ndarray) -> tuple:
            """Convert quaternion to yaw, pitch, roll angles."""
            w, x, y, z = q
            # Yaw (psi)
            psi = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
            # Pitch (theta)
            theta = np.arcsin(2*(w*y - z*x))
            # Roll (phi)
            phi = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
            return phi, theta, psi
       
        def quat_multiply(q1, q2):
            """
            Quaternion multiplication

            q = [w,x,y,z]
            """

            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2

            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])


        def crazyflie_dynamics(self, x, u):
        
            # Unpack states
            p = x[0:3]
            v = x[3:6]
            q = x[6:10]
            omega = x[10:13]
          #  omega_m = x[13:17]

            # Inputs
            thrust_cmd = u[0]
            tau_cmd = u[1:4]
            # ------------------------
            # Motor dynamics
            # ------------------------
           # omega_hover = np.sqrt(    max(thrust_cmd, 0.0)
    
            #     (4.0 * self.kF)
            # )

           # omega_m_dot = (
           #     omega_hover - omega_m
          #  ) / self.tau_m

            # ------------------------
            # Rotor forces
            # ------------------------

         #   F_i = self.kF * omega_m**2
          #  Q_i = self.kQ * omega_m**2

           # F_total = np.sum(F_i)
#
  #          tau_motor = np.array([
 #               self.d * (F_i[3] - F_i[1]),
   #             self.d * (F_i[2] - F_i[0]),
    #            Q_i[0] - Q_i[1] + Q_i[2] - Q_i[3]
     #       ])

            # Optional:
            # use commanded torques directly

       #     tau_motor += tau_cmd

            # ------------------------
            # Rotation matrix
            # ------------------------

            qw, qx, qy, qz = q

            R = np.array([
                [
                    1 - 2*qy*qy - 2*qz*qz,
                    2*qx*qy - 2*qz*qw,
                    2*qx*qz + 2*qy*qw
                ],
                [
                    2*qx*qy + 2*qz*qw,
                    1 - 2*qx*qx - 2*qz*qz,
                    2*qy*qz - 2*qx*qw
                ],
                [
                    2*qx*qz - 2*qy*qw,
                    2*qy*qz + 2*qx*qw,
                    1 - 2*qx*qx - 2*qy*qy
                ]
            ])

            # ------------------------
            # Translational dynamics
            # ------------------------

            gravity = np.array([0.0, 0.0, -self.g])
            F_total = thrust_cmd  # first order moter dynamics is not included here can be added fro futher presision 
            thrust_world = (
                R @ np.array([0.0, 0.0, F_total])
            )
            p_dot = v

            v_dot = gravity + thrust_world / self.m

       

            # ------------------------
            # Rotational dynamics
            # ------------------------

            omega_dot = np.linalg.solve(
                self.J,
                tau_cmd
                - np.cross(
                    omega,
                    self.J @ omega
                )
            )

            # ------------------------
            # Quaternion dynamics
            # ------------------------

            q_dot = 0.5 * quat_multiply(
                q,
                np.array([
                    0.0,
                    omega[0],
                    omega[1],
                    omega[2]
                ])
            )

            # ------------------------
            # Assemble x_dot
            # ------------------------

            x_dot = np.concatenate([
                p_dot,
                v_dot,
                q_dot,
                omega_dot
            ])

            return x_dot
                    
            def control_loop(self):
                pass
                    