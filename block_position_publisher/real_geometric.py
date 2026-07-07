import os
import threading
import keyboard
import rclpy
import time
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, Wrench, Vector3
from std_msgs.msg import String
from nav_msgs.msg import Odometry

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from block_position_publisher.trajectory_tracker import TrajectoryGenerator
from math import atan2, sqrt

import time

URI = 'radio://0/80/2M/E7E7E7E7E7'
connected = False

def connected_cb(link_uri):
    global connected
    print(f"Connected to {link_uri}")
    connected = True

cflib.crtp.init_drivers()
cf = Crazyflie()
cf.connected.add_callback(connected_cb)
cf.open_link(URI)

while not connected:
    time.sleep(0.1)

class InputSubscriber(Node):
    def __init__(self):
        super().__init__('input_subscriber')
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0
        self.psi = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.damping_alpha = 0.1
        self.new_telemetry_available = False
        self.vbat = 4.2
        self.setup_loggers()
        self.roll = 0.0 
        self.pitch = 0.0

        #fr z estimate
        self.z = 0.0
        self.z_recieved = False
        self.initial_pz = 0.0
        self.baseline_calibrated = False

    def setup_loggers(self):
        # High priority 10ms update group for altitude state telemetry
        self.lg_alt = LogConfig(name='Altitude', period_in_ms=10)
        self.lg_alt.add_variable('stateEstimate.ax', 'float')
        self.lg_alt.add_variable('stateEstimate.ay', 'float')
        self.lg_alt.add_variable('stateEstimate.az', 'float')
        self.lg_alt.add_variable('stabilizer.yaw', 'float')
        self.lg_alt.add_variable('stabilizer.roll', 'float')
        self.lg_alt.add_variable('stabilizer.pitch', 'float')
        cf.log.add_config(self.lg_alt)
        self.lg_alt.data_received_cb.add_callback(self.telemetry_callback)
        self.lg_alt.start()

        # Monitor battery state to scale PWM baseline on real hardware
        self.lg_bat = LogConfig(name='Battery', period_in_ms=100)
        self.lg_bat.add_variable('pm.vbat', 'float')
        self.lg_bat.add_variable('gyro.x', 'float')
        self.lg_bat.add_variable('gyro.y', 'float')
        self.lg_bat.add_variable('gyro.z', 'float')
        self.lg_bat.add_variable('stateEstimate.z', 'float')
        cf.log.add_config(self.lg_bat)
        self.lg_bat.data_received_cb.add_callback(self.battery_callback)
        self.lg_bat.start()

    def telemetry_callback(self, timestamp, data, logconf):
        if 'stateEstimate.ax' in data:
            self.ax = data['stateEstimate.ax']
        if 'stateEstimate.ay' in data:
            self.ay = data['stateEstimate.ay']
        if 'stateEstimate.az' in data:
            self.az = data['stateEstimate.az']
        if 'stabilizer.yaw' in data:
            self.psi = np.radians(data['stabilizer.yaw'])
        if 'stabilizer.roll' in data:
            self.phi = np.radians(data['stabilizer.roll'])
        if 'stabilizer.pitch' in data:
            self.theta = np.radians(data['stabilizer.pitch'])
        
        self.filtered_roll = (self.damping_alpha * self.phi) + ((1.0 - self.damping_alpha) * self.filtered_roll)
        self.filtered_pitch = (self.damping_alpha * self.theta) + ((1.0 - self.damping_alpha) * self.filtered_pitch)
        
        # Update your active state tracking variables with the clean, damped data
        self.roll = self.filtered_roll
        self.pitch = self.filtered_pitch
    
        self.new_telemetry_available = True  

    def battery_callback(self, timestamp, data, logconf):
        if 'pm.vbat' in data:
            self.vbat = data['pm.vbat']
        if 'gyro.x' in data:
            self.wx = data['gyro.x']
        if 'gyro.y' in data:
            self.wy = data['gyro.y']
        if 'gyro.z' in data:
            self.wz = data['gyro.z']
        if 'stateEstimate.z' in data:
            raw_z = data['stateEstimate.z']
            
            self.z_recieved = True
            if self.baseline_calibrated:
                self.z = raw_z - self.initial_pz
            else:
                self.z = raw_z

    def calibrate_ground_baseline(self, duration_sec=1.5):
        print("Calibrating ground zero reference...")
        z_readings = []
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            if self.z_recieved:
                z_readings.append(self.z)
            time.sleep(0.01)
        self.initial_pz = sum(z_readings) / len(z_readings) if z_readings else 0.0
        self.baseline_calibrated = True

    def get_telemetry(self):
        self.angles = np.array([self.psi, self.roll, self.pitch])
        self.omega = np.array([self.wx, self.wy, self.wz])
        self.angles=np.deg2rad(self.angles)
        self.omega=np.deg2rad(self.omega)
        return np.array([self.ax, self.ay, self.az]), self.omega, self.angles, self.vbat,self.z

    
    
    

class GeometricController(Node):
    """
    Geometric tracking controller for quadcopter.
    Based on Lee et al. 2010 - geometric tracking control on SE(3).

    Subscribes:
        /model/quadcopter/pose      → position + orientation (quaternion)
        /model/quadcopter/odometry  → linear + angular velocity

    Publishes:
        /block/force                → thrust (force.z) + torque (torque.xyz)
    """

    def __init__(self, input_subscriber):
        super().__init__('geometric_controller')
        self.input_subscriber = input_subscriber

            # ── Gains ──────────────────────────────────────────────────
        self.Kp = np.diag([4.0, 4.0, 3.78])

        self.Kd = np.diag([0.5, 0.5, 1.28])

        self.Ki = np.zeros((3,3))

        self.KR = np.diag([0.3, 0.3, 0.05])

        self.Kw = np.diag([0.05, 0.05, 0.02])
        self.log_data = []
        self.start_time=self.get_clock().now().nanoseconds / 1e9



        self.trajectory=TrajectoryGenerator(traj_type='hover',height=-1.0)
        # ── Physical parameters ────────────────────────────────────
        self.m = 0.031
        self.g = 9.81

        self.get_logger().info(f"Controller mass = {self.m}")
       
        self.J=np.array([[16.571710E-06 ,0.830806E-06,0.718277E-06],
                        [0.830806E-06,16.655602E-06,1.800197E-06],
                        [0.718277E-06,1.800197E-06,29.261652E-06]]
                        )       
        # Odometry ≈ 40 Hz
        self.dt = 0.01
        self.vbat=0
        self.p_udp_prev = 0
        self.t_udp_prev = 0

        # ── Current state (updated by subscribers) ─────────────────
        self.p     = np.array([0.0, 0.0, 0.0])                       # position
        self.q     = np.array([1.0, 0.0, 0.0, 0.0])   # quaternion [w, x, y, z]
        self.v     = np.zeros(3)                        # linear velocity
        self.omega = np.zeros(3)                        # angular velocity

        # ── Setpoints (edit here or make ROS params) ───────────────
        self.pd = np.array([0.0, 0.0, 1.0]) # desired position (hover at 1m)
        self.vd    = np.zeros(3)                  # desired velocity
        self.ad    = np.zeros(3)                  # desired acceleration
        self.psi_d = 0.0  
        self.alpha=0.2                        # desired yaw angle (rad)

        # ── Subscribers ────────────────────────────────────────────
        self.ep_int = np.zeros(3)
        
        self.udp_sub = self.create_subscription(
            Vector3,                     # Change to Float64MultiArray if your topic uses that
            '/udp_data',               # The topic name published by udp_listener.py
            self.udp_callback,
            10
        )
        self.angle=np.zeros(3)
    
        # ── Publisher ──────────────────────────────────────────────
        self.force_pub = self.create_publisher(Wrench, '/block/force', 1)

        # ── Control loop @ 1 kHz ───────────────────────────────────
        #self.create_timer(self.dt, self.control_loop)

        #self.get_logger().info('Geometric controller started. Hovering at z=1.0m')
        self.control_timer = self.create_timer(
            0.001,     # 100 Hz
            self.control_loop
        )

        self.manual_roll_deg = 0.0
        self.manual_pitch_deg = 0.0
        self.manual_yaw_rate = 0.0

        self.manual_mode = True
    # ──────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ──────────────────────────────────────────────────────────────
    def udp_callback(self, msg):

        # Current position from Vector3 message
        p_curr = np.array([msg.x, msg.y, 0])  # Assuming z is constant or obtained elsewhere

        # Store latest position
        self.p = p_curr.copy()

        # Current time
        t_curr = self.get_clock().now().nanoseconds / 1e9

        # Compute velocity using finite difference + low-pass filter
        if self.p_udp_prev is not None and self.t_udp_prev is not None:
            dt = t_curr - self.t_udp_prev

            if dt > 0.0:
                v_raw = (p_curr - self.p_udp_prev) / dt
                self.v = self.alpha * v_raw + (1 - self.alpha) * self.v

        # Update previous state
        self.p_udp_prev = p_curr.copy()
        self.t_udp_prev = t_curr

    # ──────────────────────────────────────────────────────────────
    # Math helpers
    # ──────────────────────────────────────────────────────────────

    def quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        R = np.array([
            [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w    ],
            [2*x*y + 2*z*w,          1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w    ],
            [2*x*z - 2*y*w,          2*y*z + 2*x*w,        1 - 2*x**2 - 2*y**2],
        ])
        return R
    def roll_pitch_yaw_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll, pitch, yaw angles to quaternion [w, x, y, z]."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        q = np.array([
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy   # z
        ])
        return q

    def vee(self, S: np.ndarray) -> np.ndarray:
        """Vee map: extracts the 3-vector from a skew-symmetric 3x3 matrix."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    # ──────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────
    #-----------converting omega to body frame _________________
    

    def control_loop(self):
        # ── Position & velocity errors ─────────────────────────────
        time_now = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = time_now - self.start_time


        _,self.omega,self.angles,self.vbat,self.p[2]=self.input_subscriber.get_telemetry()
        print("p =", self.p, "\n v =", self.v)
        print("angles =", self.angles,"\n omega =", self.omega)
        self.q = self.roll_pitch_yaw_to_quat(self.angles[1], self.angles[2], self.angles[0])  # roll, pitch, yawf
       
       # self.pd, self.vd, self.ad, self.psi_d = self.trajectory.get_setpoint(elapsed_time, 0.01)
      
       # print("pd =", self.pd, "\n", "vd =", self.vd, "\n ad =", self.ad, "psi_d =", self.psi_d)
        if self.p is None:
            return
        ep = self.p - self.pd        # position error
        ep=np.clip(ep, -1.0, 1.0)  # Limit position error to avoid extreme control actions
        ev = self.v - self.vd
        # ── Desired acceleration ───────────────────────────────────
        a_des = (self.ad
                 - self.Kp @ ep
                 - self.Kd @ ev
                 - self.Ki @ self.ep_int)
        a_des = np.clip(a_des, -5.0, 5.0)  # Limit desired acceleration to avoid extreme control actions
       
        # ── Desired thrust vector (world frame) ────────────────────
        #f_des = self.m * (a_des - np.array([0.0, 0.0, -self.g]))
        f_des = self.m * (a_des + np.array([0.0, 0.0, self.g]))  # gravity compensation
        print("f_des =", f_des)
        # ── Rotation matrix from current quaternion ────────────────
        R = self.quat_to_rotation_matrix(self.q)
        

        # ── Collective thrust (project onto body z-axis) ──────────
        thrust = float(f_des @ (R @ np.array([0.0, 0.0, 1.0])))
        thrust = float(np.clip(thrust, 0.0, 1.5))
        thrust_p=thrust

        # ── Desired attitude (Rd) ──────────────────────────────────
        
        b3_des = f_des / np.linalg.norm(f_des)
        b1_des = np.array([
        np.cos(self.psi_d),
        np.sin(self.psi_d),
        0.0
        ])
     
        b2_des = np.cross(b3_des, b1_des)
        b2_des = b2_des / np.linalg.norm(b2_des)

        Rd = np.column_stack([
            b1_des,
            b2_des,
            b3_des
        ])
        roll_d  = atan2( Rd[2, 1],  Rd[2, 2])
        pitch_d = atan2(-Rd[2, 0],  sqrt(Rd[2, 1]**2 + Rd[2, 2]**2))

        #---------Thrust PWM conversion----------------
        current_voltage = np.clip(self.vbat, 3.6, 4.2)

        base_multiplier = (32000- ((current_voltage - 3.6) / (4.2 - 3.6))* (32000 - 27000))

        thrust_pwm = int((thrust / (self.m * self.g)) * base_multiplier)

        thrust_pwm = int(np.clip(thrust_pwm, 10000, 1000000000000000000000000000))

        # ── Attitude error ─────────────────────────────────────────
        eR = 0.5 * self.vee(Rd.T@ R - R.T @ Rd)

        # ── Angular velocity error ─────────────────────────────────
        eomega = self.omega                          # desired omega = 0

        # ── Control torque ─────────────────────────────────────────
    
        torque = (- self.KR @ eR
                  - self.Kw @ eomega
                  + np.cross(self.omega, self.J @ self.omega))
        torque = np.clip(torque, -0.02, 0.02)
        
        #------------converter----------------
        thrust_vector = R @ np.array([0.0, 0.0, thrust]) 
      
        print("thrust_vector =", thrust_vector) # thrust in world frame
        torquer = R @ torque  # torque in world frame
        #print(thrust_vector,torquer);
        # ── Publish Wrench ─────────────────────────────────────────
        msg = Wrench()
        msg.force.x = thrust_vector[0]
        msg.force.y = thrust_vector[1]
        msg.force.z = thrust_vector[2]
        msg.torque.x = torquer[0]
        msg.torque.y = torquer[1]
        msg.torque.z = torquer[2]
        self.force_pub.publish(msg)
        
        if self.manual_mode:
            cf.commander.send_setpoint(self.manual_roll_deg,self.manual_pitch_deg,self.manual_yaw_rate,int(thrust_pwm))    # your PWM conversion)

        else:
            cf.commander.send_setpoint(np.degrees(roll_d),np.degrees(pitch_d),np.degrees(self.psi_d),int(thrust_pwm))
        
    def emergency_stop(self):
        print("\n!!! EMERGENCY CUTOUT INITIATED !!!")
        try:
            cf.commander.send_setpoint(0, 0, 0, 0)
        except:
            pass
        cf.close_link()
        os._exit(0)
        # Optional debug log (comment out for performance)
       # self.get_logger().info(f'thrust={thrust:.3f}  torque={torque}  pos={self.p}')
        # )
    
    def keyboard_flight_stick_listener(controller):
        # Maximum allowed tilt angles (Keep low for indoor testing)
        MAX_ANGLE = 8.0      
        YAW_SPEED = 30.0     

        # RAMP CONFIGURATION: Controls how fast the stick moves (in degrees per 20ms frame)
        # Higher = more snappy/twitchy, Lower = smoother/more stable
        ATTITUDE_RAMP = 0.8  
        YAW_RAMP = 3.0       

        while True:
            # Immediate Emergency Watchdog
            if keyboard.is_pressed('space'):
                controller.emergency_stop()

            # ── PITCH AXIS (W = Forward / S = Backward) ──
            if keyboard.is_pressed('w'):
                # Smoothly ramp down toward negative max (forward tilt)
                controller.manual_pitch_deg = max(controller.manual_pitch_deg - ATTITUDE_RAMP, -MAX_ANGLE)
            elif keyboard.is_pressed('s'):
                # Smoothly ramp up toward positive max (backward tilt)
                controller.manual_pitch_deg = min(controller.manual_pitch_deg + ATTITUDE_RAMP, MAX_ANGLE)
            else:
                # No key pressed: smoothly glide back to flat horizontal center (0.0)
                if controller.manual_pitch_deg > 0.1:
                    controller.manual_pitch_deg -= ATTITUDE_RAMP
                elif controller.manual_pitch_deg < -0.1:
                    controller.manual_pitch_deg += ATTITUDE_RAMP
                else:
                    controller.manual_pitch_deg = 0.0

            # ── ROLL AXIS (A = Left / D = Right) ──
            if keyboard.is_pressed('a'):
                controller.manual_roll_deg = max(controller.manual_roll_deg - ATTITUDE_RAMP, -MAX_ANGLE)
            elif keyboard.is_pressed('d'):
                controller.manual_roll_deg = min(controller.manual_roll_deg + ATTITUDE_RAMP, MAX_ANGLE)
            else:
                if controller.manual_roll_deg > 0.1:
                    controller.manual_roll_deg -= ATTITUDE_RAMP
                elif controller.manual_roll_deg < -0.1:
                    controller.manual_roll_deg += ATTITUDE_RAMP
                else:
                    controller.manual_roll_deg = 0.0

            # ── YAW AXIS (Left / Right Arrow Keys) ──
            if keyboard.is_pressed('left'):
                controller.manual_yaw_rate = max(controller.manual_yaw_rate - YAW_RAMP, -YAW_SPEED)
            elif keyboard.is_pressed('right'):
                controller.manual_yaw_rate = min(controller.manual_yaw_rate + YAW_RAMP, YAW_SPEED)
            else:
                if controller.manual_yaw_rate > 0.5:
                    controller.manual_yaw_rate -= YAW_RAMP
                elif controller.manual_yaw_rate < -0.5:
                    controller.manual_yaw_rate += YAW_RAMP
                else:
                    controller.manual_yaw_rate = 0.0

            time.sleep(0.02)


# ──────────────────────────────────────────────────────────────────
def main():
    rclpy.init()

    # Create only ONE telemetry subscriber
    input_subscriber = InputSubscriber()
    input_subscriber.calibrate_ground_baseline()  # Calibrate ground baseline before starting the controller
    # Pass it to the controller
    controller = GeometricController(input_subscriber)
    keyboard_thread = threading.Thread(
        target=controller.keyboard_flight_stick_listener,
        args=(controller,),
        daemon=True
    )
    keyboard_thread.start()
    # Spin both nodes
    executor = rclpy.executors.MultiThreadedExecutor()

    executor.add_node(input_subscriber)
    executor.add_node(controller)

    try:
        executor.spin()

    except KeyboardInterrupt:
        controller.emergency_stop()


    finally:
        input_subscriber.destroy_node()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
