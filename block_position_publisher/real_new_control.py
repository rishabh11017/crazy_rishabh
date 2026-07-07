import os
import time
import sys
import threading
import keyboard
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

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


class RealDroneController:
    def __init__(self):
        # ── GAINS FROM GAZEBO (Adapted to Real Scale) ──────────────────
        self.kp_z = 4 # Replaced old inverted logic with clean positive Gazebo gain
        self.kd_z = 2# Clean tracking derivative
        self.kp_psi = 4.0

        # Physical constants
        self.m = 0.029       
        self.g = 9.81        
        self.dt = 0.01      # 100Hz telemetry tracking window
        
        # ── DYNAMIC TARGET SETPOINTS ──────────────────────────────────
        self.pd = 0.0       # Starts at 0.0m on the ground. Use arrow keys to lift off!
        self.vd = 0.0       # Target velocity
        self.ad = 0.0       # Target acceleration
        self.psi_d = 0.0

        #filter data for k_d
        self.filter_alpha = 0.75
        
        # Historical memory states
        self.last_d_roll = 0.0
        self.last_d_pitch = 0.0
        self.manual_roll_deg = 0.0
        self.manual_pitch_deg = 0.0
        self.manual_yaw_rate = 0.0
        # ── RE-ADD SENSOR DAMPING MEMORY FIELDS ──
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        
        # Damping constant: lower means heavier smoothing/more stable (Try between 0.05 and 0.15)
        self.damping_alpha = 0.80
        # Current telemetry states
        self.pz = 0.0
        self.vz = 0.0
        self.psi = 0.0       
        self.initial_pz = 0.0
        self.initial_psi = 0.0
        self.vbat = 4.2     # Telemetry battery voltage baseline

        self.new_telemetry_available = False  # Freshness control loop gate

        # Threaded storage mapping
        self.current_setpoint = (0.0, 0.0, 0.0, 0)
        self.setpoint_lock = threading.Lock()

        self.setup_loggers()

    def setup_loggers(self):
        # High priority 10ms update group for altitude state telemetry
        self.lg_alt = LogConfig(name='Altitude', period_in_ms=10)
        self.lg_alt.add_variable('stateEstimate.z', 'float')
        self.lg_alt.add_variable('stateEstimate.vz', 'float')
        self.lg_alt.add_variable('stabilizer.yaw', 'float')
        self.lg_alt.add_variable('stabilizer.roll', 'float')
        self.lg_alt.add_variable('stabilizer.pitch', 'float')
        cf.log.add_config(self.lg_alt)
        self.lg_alt.data_received_cb.add_callback(self.telemetry_callback)
        self.lg_alt.start()

        # Monitor battery state to scale PWM baseline on real hardware
        self.lg_bat = LogConfig(name='Battery', period_in_ms=100)
        self.lg_bat.add_variable('pm.vbat', 'float')
        cf.log.add_config(self.lg_bat)
        self.lg_bat.data_received_cb.add_callback(self.battery_callback)
        self.lg_bat.start()
        
    def telemetry_callback(self, timestamp, data, logconf):
        if 'stateEstimate.z' in data:
            self.pz = data['stateEstimate.z']
        if 'stateEstimate.vz' in data:
            self.vz = data['stateEstimate.vz']
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

    def calibrate_ground_baseline(self, duration_sec=1.5):
        print("Calibrating ground zero reference and attitude tares...")
        z_readings = []
        yaw_readings = []
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            if self.pz != 0.0:
                z_readings.append(self.pz)
                yaw_readings.append(self.psi)
            time.sleep(0.01)
        self.initial_pz = sum(z_readings) / len(z_readings) if z_readings else 0.0
        self.initial_psi = sum(yaw_readings) / len(yaw_readings) if yaw_readings else 0.0
        print(f"-> Ground Baseline Set: {self.initial_pz:.3f}m")
        print(f"-> Initial Yaw Tare: {np.degrees(self.initial_psi):.1f}°")

    def z_control(self):
        """ EXTRACTED FROM YOUR GAZEBO CODE """
        relative_pz = self.pz - self.initial_pz
        
        # 1. Standard Error Framework (Target - Current)
        ep = self.pd - relative_pz
        ep = np.clip(ep, -1.0, 1.0)  
        
        # 2. Velocity Error tracking
        ev = self.vd - self.vz
        
        # 3. Pure Gazebo Acceleration Request Math
        acc_req = self.ad + (self.kp_z * ep) + (self.kd_z * ev)
        acc_req = np.clip(acc_req, -6.0, 6.0) 
        
        # 4. Total Force Output
        f_z = self.m * (self.g + acc_req)
        return np.clip(f_z, 0.01, 1.2)

    def update_setpoints(self):
        # 1. Compute Gazebo-validated vertical thrust force
        f_z = self.z_control()
        
        # 2. Dynamic Real Hardware Voltage Linear Interpolation
        current_voltage = np.clip(self.vbat, 3.6, 4.2)
        base_multiplier = 32000 - ((current_voltage - 3.6) / (4.2 - 3.6)) * (32000 - 27000)
        
        # 3. Convert Force ratio cleanly to raw PWM integer scale
        thrust_pwm = int((f_z / (self.m * self.g)) * base_multiplier)
        
        # Hardware limits safety clamp 
        thrust_pwm = np.clip(thrust_pwm, 10000, 42000)

        # ── 4. IMPLEMENT ATTITUDE STABILIZATION DAMPING ──
        # Damping gain: controls how aggressively the drone fights random drifting.
        # Try a value between 0.1 and 0.4. Too high will cause shaking.
        k_damp = 0.25  

        # ROLL AXIS: If no keyboard input, apply a counter-tilt based on filtered sensor data
        if abs(self.manual_roll_deg) > 0.01:
            roll_deg = self.manual_roll_deg
        else:
            # Active leveling: if drone rolls right (+), command a left (-) correction
            roll_deg = -k_damp * self.roll  

        # PITCH AXIS: Same principle
        if abs(self.manual_pitch_deg) > 0.01:
            pitch_deg = self.manual_pitch_deg
        else:
            # Active leveling: if drone pitches up (+), command a down (-) correction
            pitch_deg = -k_damp * self.pitch  

        # YAW RATE AXIS: Simple pass-through (No keys pressed = 0.0 rad/s rotation)
        yaw_rate_deg = np.clip(self.manual_yaw_rate, -45.0, 45.0)

        # ── 5. FINAL SAFETY CLAMP ──
        roll_deg = np.clip(roll_deg, -15.0, 15.0)
        pitch_deg = np.clip(pitch_deg, -15.0, 15.0)

        # Ship to the radio thread
        with self.setpoint_lock:
            self.current_setpoint = (roll_deg, pitch_deg, yaw_rate_deg, thrust_pwm)
            
        return thrust_pwm


def emergency_stop():
    print("\n!!! EMERGENCY CUTOUT INITIATED !!!")
    try:
        cf.commander.send_setpoint(0, 0, 0, 0)
    except:
        pass
    cf.close_link()
    os._exit(0)


def radio_commander_worker(controller, stop_event):
    """ Runs the PID math and sends commands to the drone at 50Hz """
    while not stop_event.is_set():
        # 1. Run the PID equations to update the latest setpoint values
        controller.update_setpoints()
        
        # 2. Extract those fresh values safely using the thread lock
        with controller.setpoint_lock:
            roll, pitch, yawrate, thrust = controller.current_setpoint
            
        try:
            # 3. Stream them over the radio
            cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
        except Exception:
            pass
        time.sleep(0.02)  # 50Hz frequency heartbeat


def main():
    controller = RealDroneController()
    time.sleep(3.5)  # Let Kalman filter stabilize
    controller.calibrate_ground_baseline(duration_sec=1.5)

    print("\nUnlocking pipeline safety watchdogs...")
    for _ in range(15):
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.05)
    print("System Armed. Operational.")
    print("-----------------------------------------------------------------")
    print("Type a target height in meters (e.g., 0.5) and press ENTER.")
    print("HOTKEY: Press [SPACEBAR] to execute immediate emergency stop.")
    print("-----------------------------------------------------------------")

    # Fire up the background 50Hz controller loop (Now handles PID math + Radio transmitting)
    stop_radio_thread = threading.Event()
    radio_thread = threading.Thread(target=radio_commander_worker, args=(controller, stop_radio_thread), daemon=True)
    radio_thread.start()

    # Safety hotkey shortcut listener calling original emergency_stop
    # def keyboard_shortcut_listener():
    #     while True:
    #         if keyboard.is_pressed('space') or keyboard.is_pressed('q'):
    #             emergency_stop()
    #         time.sleep(0.02)
    # 2. Background Keyboard Flight Stick + Emergency Listener Loop
    def keyboard_flight_stick_listener():
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
                emergency_stop()

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

            time.sleep(0.02)  # Maintain stable 50Hz polling frequency1
            

    flight_stick_thread = threading.Thread(target=keyboard_flight_stick_listener, daemon=True)
    flight_stick_thread.start()

    # Handle standard text entries on the main thread loop
    try:
        while True:
            # Displays state exactly once right before prompting you
            relative_height = controller.pz - controller.initial_pz
            print(f"\n[Current State] Alt: {relative_height:.2f}m | Target: {controller.pd:.2f}m")
            
            user_input = input("Enter Target Height (m) > ").strip().lower()
            if not user_input:
                continue

            try:
                target_height = float(user_input)
                if target_height < 0.0:
                    print("Error: Target height cannot be negative.")
                elif target_height >5.0:
                    print("Safety Warning: Target height capped at 15.0 meters.")
                    controller.pd = 5.0
                else:
                    controller.pd = round(target_height, 2)
                    print(f"Target height updated! Command dispatched to climb to: {controller.pd}m")
            except ValueError:
                print("Invalid input. Type a valid height number, or hit Spacebar to stop.")
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting via Terminal Break Sequence.")
    finally:
        stop_radio_thread.set()
        emergency_stop()


if __name__ == "__main__":
    main() 