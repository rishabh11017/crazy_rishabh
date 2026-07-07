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
        self.hover_pos  = np.array(kwargs.get('hover_pos',  [0.0, 0.0, 8.5]))

        # ── Circle ────────────────────────────────────────────────
        self.radius     = kwargs.get('radius',   1.0)   # m
        self.omega_traj = kwargs.get('omega',    0.2)   # rad/s
        self.altitude   = kwargs.get('altitude', 3.5)   # m
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
