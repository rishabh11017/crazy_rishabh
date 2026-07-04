"""
Trajectory Tracking Controller for Quadcopter — ROS2 / Python
==============================================================
Architecture: Cascaded PID (outer position loop → inner attitude extraction)

Sensors used (subscribed topics):
  /model/quadcopter/pose       PoseStamped   → px, py, pz + orientation (quat)
  /model/quadcopter/odometry   Odometry      → vx, vy, vz + omega_x/y/z
  /imu                         Imu           → ax, ay, az + omega (backup)

Command output (published topic):
  /drone/cmd_attitude          Float64MultiArray → [thrust, roll, pitch, yaw]

Trajectory types (set via ROS2 param 'trajectory_type'):
  hover     — hold a fixed position
  circle    — horizontal circle at fixed altitude
  helix     — ascending circle
  figure8   — lemniscate (figure-of-8)
  waypoints — follow a list of waypoints in sequence

Author : generated for Black Hornet digital twin project
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import numpy as np
from math import sin, cos, atan2, sqrt, asin, pi

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Wrench
#timefrom zero




# ══════════════════════════════════════════════════════════════════════════════
# Trajectory Generator
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryGenerator:
    """
    Generates desired position, velocity, and acceleration at time t.
    Returns: pd (3,), vd (3,), ad (3,), yaw_d (float)
    """

    def __init__(self, traj_type: str = 'hover', **kwargs):
        self.traj_type = traj_type

        # ── Hover ──────────────────────────────────────────────────
        self.hover_pos  = np.array(kwargs.get('hover_pos',  [0.0, 0.0, 1.5]))

        # ── Circle ────────────────────────────────────────────────
        self.radius     = kwargs.get('radius',   1.0)   # m
        self.omega_traj = kwargs.get('omega',    0.2)   # rad/s
        self.altitude   = kwargs.get('altitude', 1.5)   # m
        self.center     = np.array(kwargs.get('center', [0.0, 0.0]))

        # ── Helix (circle + vertical climb) ───────────────────────
        self.climb_rate = kwargs.get('climb_rate', 0.1)  # m/s

        # ── Waypoints ─────────────────────────────────────────────
        # Each waypoint: [x, y, z, yaw_deg, hold_time_s]
        self.waypoints  = kwargs.get('waypoints', [
            [0.0,  0.0,  1.0, 0.0,  3.0],
            [2.0,  0.0,  1.5, 0.0,  3.0],
            [2.0,  2.0,  1.5, 90.0, 3.0],
            [0.0,  2.0,  1.0, 180.0,3.0],
            [0.0,  0.0,  1.0, 0.0,  3.0],
        ])
        self._wp_index  = 0
        self._wp_timer  = 0.0

        self.yaw_d      = 0.0   # default desired yaw

    def get_setpoint(self, t: float, dt: float):
        """Return (pd, vd, ad, yaw_d) at time t."""

        if self.traj_type == 'hover':
            return self._hover()

        elif self.traj_type == 'circle':
            return self._circle(t)

        elif self.traj_type == 'helix':
            return self._helix(t)

        elif self.traj_type == 'figure8':
            return self._figure8(t)

        elif self.traj_type == 'waypoints':
            return self._waypoints(dt)

        else:
            return self._hover()

    # ── Individual trajectory implementations ──────────────────────────────

    def _hover(self):
        pd  = self.hover_pos.copy()
        vd  = np.zeros(3)
        ad  = np.zeros(3)
        return pd, vd, ad, self.yaw_d

    def _circle(self, t):
        w = self.omega_traj
        R = self.radius
        cx, cy = self.center

        pd = np.array([
            cx + R * cos(w * t),
            cy + R * sin(w * t),
            self.altitude
        ])
        vd = np.array([
            -R * w * sin(w * t),
             R * w * cos(w * t),
             0.0
        ])
        ad = np.array([
            -R * w**2 * cos(w * t),
            -R * w**2 * sin(w * t),
             0.0
        ])
        # yaw tracks the velocity direction
        yaw_d = 0
        return pd, vd, ad, yaw_d

    def _helix(self, t):
        pd, vd, ad, yaw_d = self._circle(t)
        pd[2]  = self.altitude + self.climb_rate * t
        vd[2]  = self.climb_rate
        ad[2]  = 0.0
        return pd, vd, ad, yaw_d

    def _figure8(self, t):
        """Lemniscate of Bernoulli — figure-of-8 in XY plane."""
        w = self.omega_traj
        R = self.radius
        cx, cy = self.center

        # parametric equations
        denom  = 1 + sin(w * t)**2

        pd = np.array([
            cx + R * cos(w * t) / denom,
            cy + R * sin(w * t) * cos(w * t) / denom,
            self.altitude
        ])

        # numerical velocity (finite diff would also work here)
        dt_small = 1e-2
        t2 = t + dt_small
        d2 = 1 + sin(w * t2)**2
        pd2 = np.array([
            cx + R * cos(w * t2) / d2,
            cy + R * sin(w * t2) * cos(w * t2) / d2,
            self.altitude
        ])
        vd = (pd2 - pd) / dt_small

        # numerical acceleration
        t0 = t - dt_small
        d0 = 1 + sin(w * t0)**2
        pd0 = np.array([
            cx + R * cos(w * t0) / d0,
            cy + R * sin(w * t0) * cos(w * t0) / d0,
            self.altitude
        ])
        ad = (pd2 - 2*pd + pd0) / dt_small**2

        yaw_d = 0
        return pd, vd, ad, yaw_d

    def _waypoints(self, dt: float):
        if self._wp_index >= len(self.waypoints):
            # hold last waypoint
            wp  = self.waypoints[-1]
            pd  = np.array(wp[:3])
            return pd, np.zeros(3), np.zeros(3), np.deg2rad(wp[3])

        wp          = self.waypoints[self._wp_index]
        pd          = np.array(wp[:3])
        hold_time   = wp[4]
        yaw_d       = 0

        self._wp_timer += dt
        if self._wp_timer >= hold_time:
            self._wp_timer  = 0.0
            self._wp_index += 1

        return pd, np.zeros(3), np.zeros(3), yaw_d


# ══════════════════════════════════════════════════════════════════════════════
# PID Position Controller → Attitude Commands
# ══════════════════════════════════════════════════════════════════════════════

class PositionController:
    """
    Cascaded position PID → desired attitude (roll, pitch) + thrust + yaw.

    Step 1: PID on position error → desired acceleration
    Step 2: Geometric mapping     → desired attitude + thrust
    """

    def __init__(self):
        # ── Gains ──────────────────────────────────────────────────
        self.Kp = np.diag([4.0,  4.0,  5.0 ])   # position proportional
        self.Kd = np.diag([3.0,  3.0,  4.0 ])   # velocity derivative
        self.Ki = np.diag([0.0,  0.0,  0.0])   # integral

        # ── Physical params ────────────────────────────────────────
        self.m   = 0.031    # mass kg (tune to your drone)
        self.g   = 9.81     # gravity m/s²

        # ── Integral state ─────────────────────────────────────────
        self.ep_int      = np.zeros(3)
        self.int_limit   = 2.0    # anti-windup clamp (m·s)

        # ── Output limits ──────────────────────────────────────────
        self.thrust_min  = 0.0
        self.thrust_max  = 1.5    # normalised (0 → 1.5 N range)
        self.angle_limit = np.deg2rad(30)  # ±30° max tilt

    def compute(self,
                p:     np.ndarray,   # current position    (3,)
                v:     np.ndarray,   # current velocity    (3,)
                q:     np.ndarray,   # current quaternion  [w,x,y,z]
                omega: np.ndarray,   # current ang-vel     (3,)
                pd:    np.ndarray,   # desired position    (3,)
                vd:    np.ndarray,   # desired velocity    (3,)
                ad:    np.ndarray,   # desired accel (ff)  (3,)
                yaw_d: float,        # desired yaw         (rad)
                dt:    float):       # timestep            (s)
        """
        Returns: thrust (float), roll_d (float), pitch_d (float), yaw_d (float)
        """

        # ── Position & velocity errors ─────────────────────────────
        ep = p - pd
        ev = v - vd

        # ── Integral with anti-windup ──────────────────────────────
        self.ep_int += ep * dt
        self.ep_int  = np.clip(self.ep_int, -self.int_limit, self.int_limit)

        # ── Desired acceleration (PID + feedforward) ───────────────
        a_des = (ad
                 - self.Kp @ ep
                 - self.Kd @ ev
                 - self.Ki @ self.ep_int)

        # ── Desired thrust vector (world frame) ────────────────────
        f_des    = self.m * (a_des + np.array([0.0, 0.0, self.g]))
        f_des_z  = max(f_des[2], 0.1 * self.m * self.g)  # prevent negative thrust

        # ── Collective thrust (scalar) ─────────────────────────────
        thrust   = np.linalg.norm(f_des)
        thrust   = float(np.clip(thrust, self.thrust_min, self.thrust_max))

        # ── Desired attitude from geometric mapping ────────────────
        b3_des   = f_des / np.linalg.norm(f_des)          # body-z direction

        b1_c     = np.array([cos(yaw_d), sin(yaw_d), 0.0])
        b2_cross = np.cross(b3_des, b1_c)
        norm_b2  = np.linalg.norm(b2_cross)

        if norm_b2 < 1e-6:
            # singularity guard — vertical desired heading
            b2_des = np.array([-sin(yaw_d), cos(yaw_d), 0.0])
        else:
            b2_des = b2_cross / norm_b2

        b1_des = np.cross(b2_des, b3_des)
        Rd     = np.column_stack([b1_des, b2_des, b3_des])  # desired rotation matrix

        # ── Extract Euler angles from Rd ───────────────────────────
        roll_d  = atan2( Rd[2, 1],  Rd[2, 2])
        pitch_d = atan2(-Rd[2, 0],  sqrt(Rd[2, 1]**2 + Rd[2, 2]**2))
        # yaw is commanded directly (yaw_d passed in)

        # ── Clamp attitude commands ────────────────────────────────
        roll_d  = float(np.clip(roll_d,  -self.angle_limit, self.angle_limit))
        pitch_d = float(np.clip(pitch_d, -self.angle_limit, self.angle_limit))

        return thrust, roll_d, pitch_d, yaw_d


# ══════════════════════════════════════════════════════════════════════════════
# Utility — Quaternion to Euler
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_euler(q: np.ndarray):
    """
    Convert quaternion [w, x, y, z] → (roll, pitch, yaw) in radians.
    Uses ZYX (aerospace) convention.
    """
    w, x, y, z = q

    # Roll (x-axis)
    sinr_cosp =  2.0 * (w*x + y*z)
    cosr_cosp =  1.0 - 2.0 * (x*x + y*y)
    roll      = atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp      =  2.0 * (w*y - z*x)
    sinp      = np.clip(sinp, -1.0, 1.0)
    pitch     = asin(sinp)

    # Yaw (z-axis)
    siny_cosp =  2.0 * (w*z + x*y)
    cosy_cosp =  1.0 - 2.0 * (y*y + z*z)
    yaw       = atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryTrackerNode(Node):
    """
    ROS2 node that wires the trajectory generator + position controller
    into a real-time control loop.

    Published topic:
        /drone/cmd_attitude   Float64MultiArray  [thrust, roll, pitch, yaw]
        index:                                    [0]      [1]   [2]    [3]

    Units:
        thrust  — normalised (0 … 1.5)
        roll    — radians
        pitch   — radians
        yaw     — radians
    """

    def __init__(self):
        super().__init__('trajectory_tracker')

        # ── ROS2 parameters (override from launch file / CLI) ──────
        self.declare_parameter('trajectory_type', 'circle')
        self.declare_parameter('radius',           1.0)
        self.declare_parameter('omega',            0.4)
        self.declare_parameter('altitude',         1.5)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('mass',             0.031)

        traj_type   = self.get_parameter('trajectory_type').value
        radius      = self.get_parameter('radius').value
        omega       = self.get_parameter('omega').value
        altitude    = self.get_parameter('altitude').value
        ctrl_hz     = self.get_parameter('control_rate_hz').value
        mass        = self.get_parameter('mass').value

        self.dt     = 0.1 / ctrl_hz
        self.t      = 0.0          # elapsed time since node start

        # ── Sub-systems ────────────────────────────────────────────
        self.traj   = TrajectoryGenerator(
            traj_type   = traj_type,
            radius      = radius,
            omega       = omega,
            altitude    = altitude,
        )
        self.ctrl       = PositionController()
        self.ctrl.m     = mass

        # ── Sensor state (updated by subscribers) ──────────────────
        self.p          = np.zeros(3)
        self.v          = np.zeros(3)
        self.a          = np.zeros(3)
        self.q          = np.array([1.0, 0.0, 0.0, 0.0])   # [w,x,y,z]
        self.omega      = np.zeros(3)
        self.roll       = 0.0
        self.pitch      = 0.0
        self.yaw        = 0.0

        self._got_pose  = False
        self._got_odom  = False

        # ── Subscribers ────────────────────────────────────────────
       
        self.create_subscription(
            Odometry,
            '/model/quadcopter/odometry',
            self._odom_cb,
            10
        )
        self.create_subscription(
            Imu,
            '/imu',
            self._imu_cb,
            10
        )

        # ── Publisher ──────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/drone/cmd_attitude',
            10
        )
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.force_pub = self.create_publisher(Wrench, '/block/force', 1)

        # ── Control loop timer ─────────────────────────────────────
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f'TrajectoryTracker started | '
            f'type={traj_type} | '
            f'radius={radius}m | '
            f'omega={omega}rad/s | '
            f'alt={altitude}m | '
            f'rate={ctrl_hz}Hz'
        )

    # ── Subscriber callbacks ───────────────────────────────────────────────
    def quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def _odom_cb(self, msg: Odometry):
        """Extract linear + angular velocity from /model/quadcopter/odometry."""
        self.p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        self.q = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,        
    msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ])
        self.v = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])
        self.omega = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ])
        self._got_odom = True

    def _imu_cb(self, msg: Imu):
        """Extract linear acceleration from IMU (backup / additional sensor)."""
        self.a = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        # IMU also gives angular velocity — overwrite if odom not available
        if not self._got_odom:
            self.omega = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])

    # ── Main control loop ──────────────────────────────────────────────────

    def _control_loop(self):
        # Wait until at least pose is received
       

        # ── Get trajectory setpoint ────────────────────────────────
        pd, vd, ad, yaw_d = self.traj.get_setpoint(self.t, self.dt)

        # ── Run position controller ────────────────────────────────
        thrust, roll_d, pitch_d, yaw_d = self.ctrl.compute(
            p     = self.p,
            v     = self.v,
            q     = self.q,
            omega = self.omega,
            pd    = pd,
            vd    = vd,
            ad    = ad,
            yaw_d = yaw_d,
            dt    = self.dt
        )

        # ── Publish command ────────────────────────────────────────
        forces=np.array([0,0,thrust])
        
        forces=self.quat_to_rotation_matrix(self.q) @ forces
        wrench_msg = Wrench()
        wrench_msg.force.x = forces[0]
        wrench_msg.force.y = forces[1]
        wrench_msg.force.z = forces[2]
        self.force_pub.publish(wrench_msg)
        msg = Twist()
        msg.angular.x = roll_d
        msg.angular.y = pitch_d
        msg.angular.z = yaw_d
        self.pub.publish(msg)
        print("thrust: ", thrust, "roll: ", roll_d, "pitch: ", pitch_d, "yaw: ", yaw_d)
        
        


       
        # ── Advance time ───────────────────────────────────────────
        self.t += self.dt

        # ── Debug log (1 Hz) ───────────────────────────────────────
        self.get_logger().info(
            f'[t={self.t:.1f}s] '
            f'pos=({self.p[0]:.2f},{self.p[1]:.2f},{self.p[2]:.2f}) '
            f'des=({pd[0]:.2f},{pd[1]:.2f},{pd[2]:.2f}) | '
            f'cmd: T={thrust:.3f} R={np.rad2deg(roll_d):.1f}° '
            f'P={np.rad2deg(pitch_d):.1f}° Y={np.rad2deg(yaw_d):.1f}°',
            throttle_duration_sec=1.0
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = TrajectoryTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
