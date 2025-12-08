#!/usr/bin/env python3
"""
Helios Leg (LW-360 Gen2V1) MuJoCo Deployment Script

Biped wheeled-leg robot MuJoCo simulation for validating Isaac Lab trained policies.

Joint Structure (6 DOF):
- 4 leg joints (position control): right_thigh, right_calf, left_thigh, left_calf
- 2 wheel joints (velocity control): right_foot, left_foot

Observation Dimension (27D):
- base_ang_vel: 3 (angular velocity)
- projected_gravity: 3
- velocity_commands: 3
- joint_pos: 6 (wheel positions = 0)
- joint_vel: 6
- actions: 6 (previous action)

Action Dimension (6D):
- Leg position control (4D): right_thigh, right_calf, left_thigh, left_calf
- Wheel velocity control (2D): right_foot, left_foot

Usage:
    python sim_leg.py --policy-path <policy_path> [--keyboard]
"""

import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from pynput import keyboard as pynput_keyboard
import threading
import time

# ============================================================================
# DOF Configuration - Helios Leg joint index mapping
# ============================================================================
# MuJoCo 中关节按 XML body 嵌套顺序排列:
#   qpos[0:7]  = floating_base (pos xyz + quat wxyz)
#   qpos[7]    = right_thigh_joint
#   qpos[8]    = right_calf_joint
#   qpos[9]    = right_foot_joint (轮子)
#   qpos[10]   = left_thigh_joint
#   qpos[11]   = left_calf_joint
#   qpos[12]   = left_foot_joint (轮子)
#
# 代码期望的顺序 (与 actuator 和 joint_names 一致):
#   [0] right_thigh, [1] right_calf, [2] left_thigh, [3] left_calf,
#   [4] right_foot, [5] left_foot

# MuJoCo joint position indices (qpos) - 重新映射到期望顺序
# 期望顺序: right_thigh, right_calf, left_thigh, left_calf, right_foot, left_foot
dof_ids = [7, 8, 10, 11, 9, 12]

# MuJoCo joint velocity indices (qvel) - qvel 索引 = qpos 索引 - 1 (因为 free joint 有 7 qpos 但 6 qvel)
# qvel[0:6]  = floating_base (vel xyz + ang_vel xyz)
# qvel[6]    = right_thigh_joint
# qvel[7]    = right_calf_joint
# qvel[8]    = right_foot_joint
# qvel[9]    = left_thigh_joint
# qvel[10]   = left_calf_joint
# qvel[11]   = left_foot_joint
dof_vel = [6, 7, 9, 10, 8, 11]


class SimConfig:
    """Simulation configuration"""
    def __init__(self, dt=0.005, decimation=4, sim_duration=60.0):
        self.sim_duration = sim_duration  # Simulation duration [s]
        self.dt = dt                      # Physics timestep [s]
        self.decimation = decimation      # Control frequency decimation


class RobotConfig:
    """Helios Leg robot configuration"""
    def __init__(self):
        # Joint names (order matches action order)
        self.joint_names = [
            "right_thigh_joint", "right_calf_joint",
            "left_thigh_joint", "left_calf_joint",
            "right_foot_joint", "left_foot_joint"
        ]

        # PD controller stiffness Kp [Nm/rad] - 与 helios_leg.py 匹配
        self.kp = {
            "right_thigh_joint": 100.0, "left_thigh_joint": 100.0,
            "right_calf_joint": 100.0, "left_calf_joint": 100.0,
            "right_foot_joint": 0.0, "left_foot_joint": 0.0,  # Wheels use velocity control, stiffness=0
        }

        # PD controller damping Kd [Nm*s/rad] - 与 helios_leg.py 匹配
        self.kd = {
            "right_thigh_joint": 5.0, "left_thigh_joint": 5.0,
            "right_calf_joint": 5.0, "left_calf_joint": 5.0,
            "right_foot_joint": 2.0, "left_foot_joint": 2.0,
        }

        # Integral gain Ki [Nm/(rad*s)]
        self.ki = {
            "right_thigh_joint": 2.0, "left_thigh_joint": 2.0,
            "right_calf_joint": 2.0, "left_calf_joint": 2.0,
            "right_foot_joint": 0.0, "left_foot_joint": 0.0,
        }

        # Integral saturation limit [rad*s]
        self.integral_limit = {
            "right_thigh_joint": 0.5, "left_thigh_joint": 0.5,
            "right_calf_joint": 0.5, "left_calf_joint": 0.5,
            "right_foot_joint": 0.0, "left_foot_joint": 0.0,
        }

        # Torque limit [Nm]
        self.tau_limit = {
            "right_thigh_joint": 40.0, "left_thigh_joint": 40.0,
            "right_calf_joint": 40.0, "left_calf_joint": 40.0,
            "right_foot_joint": 40.0, "left_foot_joint": 40.0,
        }

        # Action scaling (must match training config!)
        # 关节动作缩放需分别对齐 Isaac Lab 中的大腿/小腿配置
        self.leg_scale = np.array([0.125, 0.25, 0.125, 0.25], dtype=np.float32)
        self.wheel_scale = 5.0       # Wheel joints

        # Initial height [m]
        self.init_height = 0.32

        # Convert to arrays (ordered by joint_names)
        self.kp_array = np.array([self.kp[name] for name in self.joint_names])
        self.kd_array = np.array([self.kd[name] for name in self.joint_names])
        self.ki_array = np.array([self.ki[name] for name in self.joint_names])
        self.integral_limit_array = np.array([self.integral_limit[name] for name in self.joint_names])
        self.tau_limit_array = np.array([self.tau_limit[name] for name in self.joint_names])

    def print_parameters(self):
        """Print all PID control parameters"""
        print("\n" + "="*80)
        print("Helios Leg PID Control Parameters")
        print("="*80)
        print(f"{'Joint Name':<22} {'Kp':>10} {'Kd':>10} {'Ki':>10} {'tau_limit':>10}")
        print(f"{'':22} {'[Nm/rad]':>10} {'[Nm*s/rad]':>10} {'':>10} {'[Nm]':>10}")
        print("-"*80)
        for name in self.joint_names:
            print(f"{name:<22} {self.kp[name]:>10.1f} {self.kd[name]:>10.1f} {self.ki[name]:>10.1f} {self.tau_limit[name]:>10.1f}")
        print("="*80 + "\n")


class Config:
    def __init__(self):
        self.sim_config = SimConfig()
        self.robot_config = RobotConfig()


cfg = Config()

# Default joint angles (standing pose within joint limits)
# right_thigh: [0.7, 2.7], right_calf: [-2.1, 0.9]
# left_thigh: [-2.7, -0.7], left_calf: [-0.9, 2.1]
default_joint_angles = {
    "right_thigh_joint":1.7,    # Thigh initial angle
    "right_calf_joint": -1.2,    # Calf initial angle
    "left_thigh_joint": -1.7,    # Thigh initial angle
    "left_calf_joint": 1.2,      # Calf initial angle
    "right_foot_joint": 0.0,     # Wheel initial position
    "left_foot_joint": 0.0,      # Wheel initial position
}

default_angle = np.array([
    default_joint_angles["right_thigh_joint"],
    default_joint_angles["right_calf_joint"],
    default_joint_angles["left_thigh_joint"],
    default_joint_angles["left_calf_joint"],
    default_joint_angles["right_foot_joint"],
    default_joint_angles["left_foot_joint"],
], dtype=np.double)


class PDTuner:
    """Real-time PD parameter tuner (grouped by joint type)"""
    def __init__(self, robot_config):
        self.cfg = robot_config
        self.joint_names = robot_config.joint_names

        # Joint type groups
        self.joint_types = ["thigh", "calf", "foot"]
        self.selected_type = 0

        # Adjustment step sizes
        self.kp_step = 5.0
        self.kd_step = 0.5
        self.ki_step = 0.5

        # Joint type to indices mapping
        self.type_to_indices = {
            "thigh": [0, 2],  # right_thigh, left_thigh
            "calf": [1, 3],   # right_calf, left_calf
            "foot": [4, 5]    # right_foot, left_foot
        }

    def select_next_type(self):
        """Switch to next joint type"""
        self.selected_type = (self.selected_type + 1) % len(self.joint_types)
        self._print_status()

    def select_prev_type(self):
        """Switch to previous joint type"""
        self.selected_type = (self.selected_type - 1) % len(self.joint_types)
        self._print_status()

    def increase_kp(self):
        """Increase Kp for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kp[name] += self.kp_step
            self.cfg.kp_array[idx] += self.kp_step
        self._print_status()

    def decrease_kp(self):
        """Decrease Kp for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kp[name] = max(0, self.cfg.kp[name] - self.kp_step)
            self.cfg.kp_array[idx] = max(0, self.cfg.kp_array[idx] - self.kp_step)
        self._print_status()

    def increase_kd(self):
        """Increase Kd for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kd[name] += self.kd_step
            self.cfg.kd_array[idx] += self.kd_step
        self._print_status()

    def decrease_kd(self):
        """Decrease Kd for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kd[name] = max(0, self.cfg.kd[name] - self.kd_step)
            self.cfg.kd_array[idx] = max(0, self.cfg.kd_array[idx] - self.kd_step)
        self._print_status()

    def increase_ki(self):
        """Increase Ki for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.ki[name] += self.ki_step
            self.cfg.ki_array[idx] += self.ki_step
        self._print_status()

    def decrease_ki(self):
        """Decrease Ki for all joints of selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.ki[name] = max(0, self.cfg.ki[name] - self.ki_step)
            self.cfg.ki_array[idx] = max(0, self.cfg.ki_array[idx] - self.ki_step)
        self._print_status()

    def _print_status(self):
        """Print parameters for selected type"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]

        print(f"\n=== Selected Type: {joint_type.upper()} ===")
        for idx in indices:
            name = self.joint_names[idx]
            print(f"  [{idx}] {name:<22} Kp={self.cfg.kp[name]:>7.1f}  Kd={self.cfg.kd[name]:>6.1f}  Ki={self.cfg.ki[name]:>6.1f}")

    def print_all_params(self):
        """Print all joint PID parameters"""
        print("\n" + "="*70)
        print("Current PID Parameters (by Joint Type)")
        print("="*70)

        for joint_type in self.joint_types:
            indices = self.type_to_indices[joint_type]
            marker = " *SELECTED*" if joint_type == self.joint_types[self.selected_type] else ""
            print(f"\n{joint_type.upper()}{marker}:")
            print(f"  {'Idx':<4} {'Joint Name':<22} {'Kp':>10} {'Kd':>10} {'Ki':>10}")
            print("  " + "-"*58)
            for idx in indices:
                name = self.joint_names[idx]
                print(f"  {idx:<4} {name:<22} {self.cfg.kp[name]:>10.1f} {self.cfg.kd[name]:>10.1f} {self.cfg.ki[name]:>10.1f}")
        print("="*70 + "\n")


# Global PDTuner instance
pd_tuner = None


class Cmd:
    """Velocity command"""
    def __init__(self):
        # Helios Leg primarily moves in Y direction (lateral)
        self.vx = 0.0    # X direction (forward/backward) - may not work for this robot
        self.vy = 0.0    # Y direction (lateral)
        self.dyaw = 0.0  # Rotation

        self.vx_step = 0.1
        self.vy_step = 0.1
        self.vyaw_step = 0.1
        self.vx_max = 1.0
        self.vy_max = 1.0
        self.vyaw_max = 1.0

    def increase_vx(self):
        self.vx = min(self.vx + self.vx_step, self.vx_max)

    def decrease_vx(self):
        self.vx = max(self.vx - self.vx_step, -self.vx_max)

    def increase_vy(self):
        self.vy = min(self.vy + self.vy_step, self.vy_max)

    def decrease_vy(self):
        self.vy = max(self.vy - self.vy_step, -self.vy_max)

    def increase_vyaw(self):
        self.dyaw = min(self.dyaw + self.vyaw_step, self.vyaw_max)

    def decrease_vyaw(self):
        self.dyaw = max(self.dyaw - self.vyaw_step, -self.vyaw_max)

    def stop_all(self):
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0

    def stop_vx(self):
        self.vx = 0.0

    def stop_vy(self):
        self.vy = 0.0

    def stop_vyaw(self):
        self.dyaw = 0.0


vel_cmd = Cmd()


# Control mode state machine
class ControlMode:
    NORMAL = 0       # Policy control (default)
    FROZEN = 1       # Freeze target position
    INTERPOLATING = 2  # Smooth interpolation to preset pose


class PoseController:
    """Manages smooth interpolation to preset poses"""
    def __init__(self):
        # Preset poses (4 leg joints)
        # Order: right_thigh, right_calf, left_thigh, left_calf
        self.preset_pose_1 = np.array([
            1.7, -1.2,   # Right leg
            -1.7, 1.2    # Left leg
        ], dtype=np.double)

        self.preset_pose_2 = np.array([
            2.0, -0.5,   # Right leg - more upright
            -2.0, 0.5    # Left leg
        ], dtype=np.double)

        # Joint motion speed [rad/s]
        self.joint_speed = 0.5

        # Control mode state
        self.mode = ControlMode.NORMAL
        self.current_stage = 0

        # Interpolation state
        self.start_q = None
        self.target_q_preset = None
        self.interpolation_progress = 0.0
        self.interpolation_duration = None

    def start_interpolation(self, current_q, target_pose):
        """Start smooth interpolation from current to target pose"""
        self.mode = ControlMode.INTERPOLATING
        self.start_q = current_q[:4].copy()  # Only leg joints
        self.target_q_preset = target_pose.copy()

        # Calculate interpolation duration based on max displacement
        max_displacement = np.max(np.abs(self.target_q_preset - self.start_q))
        self.interpolation_duration = max_displacement / self.joint_speed
        self.interpolation_progress = 0.0

        pose_name = "Pose 1" if self.current_stage == 1 else "Pose 2"
        print(f"\n> Starting interpolation to {pose_name} (duration: {self.interpolation_duration:.2f}s)")
        print(f"  Max joint displacement: {max_displacement:.3f} rad")
        print(f"  Joint speed: {self.joint_speed:.2f} rad/s")

    def update_interpolation(self, dt):
        """Update interpolation progress and return interpolated target"""
        if self.mode != ControlMode.INTERPOLATING:
            return None

        self.interpolation_progress += dt

        # Compute interpolation factor (0 to 1)
        alpha = min(1.0, self.interpolation_progress / self.interpolation_duration)

        # Linear interpolation
        interpolated_q = self.start_q + alpha * (self.target_q_preset - self.start_q)

        # Print progress every 0.5 seconds
        if int(self.interpolation_progress * 2) > int((self.interpolation_progress - dt) * 2):
            remaining = self.interpolation_duration - self.interpolation_progress
            print(f"  Progress: {alpha*100:.1f}% | Remaining: {remaining:.2f}s")

        # Check if interpolation is complete
        if alpha >= 1.0:
            self.mode = ControlMode.FROZEN

            if self.current_stage == 1:
                self.current_stage = 2
                print("[OK] Reached Pose 1 - pose locked")
                print("  Press Space again to interpolate to Pose 2")
            elif self.current_stage == 2:
                self.current_stage = 3
                print("[OK] Reached Pose 2 - pose locked")
                print("  Press Space again to resume normal policy control")

        return interpolated_q

    def toggle_mode(self, current_q):
        """Toggle control mode"""
        if self.mode == ControlMode.NORMAL:
            self.mode = ControlMode.FROZEN
            self.current_stage = 1
            print("\n[PAUSE] [1st press] Control FROZEN - target_q locked")
            print("  Press Space again to start interpolation to Pose 1")

        elif self.mode == ControlMode.FROZEN:
            if self.current_stage == 1:
                print("\n> [2nd press] Starting interpolation to Pose 1...")
                self.start_interpolation(current_q, self.preset_pose_1)

            elif self.current_stage == 2:
                print("\n> [3rd press] Starting interpolation to Pose 2...")
                self.start_interpolation(current_q, self.preset_pose_2)

            elif self.current_stage == 3:
                self.mode = ControlMode.NORMAL
                self.current_stage = 0
                print("\n> [4th press] Resuming NORMAL policy control")

        elif self.mode == ControlMode.INTERPOLATING:
            self.mode = ControlMode.NORMAL
            self.current_stage = 0
            print("\n[STOP] Interpolation aborted - resuming NORMAL policy control")

    def increase_speed(self):
        """Increase joint motion speed"""
        self.joint_speed = min(5.0, self.joint_speed + 0.1)
        print(f"\n[FAST] Joint speed increased: {self.joint_speed:.2f} rad/s")

    def decrease_speed(self):
        """Decrease joint motion speed"""
        self.joint_speed = max(0.1, self.joint_speed - 0.1)
        print(f"\n[SLOW] Joint speed decreased: {self.joint_speed:.2f} rad/s")

    def print_preset_pose(self):
        """Print preset pose configurations"""
        joint_names = ["right_thigh", "right_calf", "left_thigh", "left_calf"]

        print("\n" + "="*70)
        print("Helios Leg Preset Pose Configurations")
        print("="*70)
        print(f"{'Joint Name':<15} {'Pose 1 [rad]':>15} {'Pose 2 [rad]':>15}")
        print("-"*70)
        for i, name in enumerate(joint_names):
            print(f"{name:<15} {self.preset_pose_1[i]:>15.3f} {self.preset_pose_2[i]:>15.3f}")
        print("-"*70)
        print(f"Joint Speed: {self.joint_speed} rad/s")
        print("="*70 + "\n")


# Global pose controller
pose_ctrl = PoseController()

# Global variable to store current joint positions
current_q_global = None


def start_keyboard_listener():
    """Start keyboard listener"""
    from pynput import keyboard as pynput_keyboard

    def on_press(key):
        global pd_tuner, pose_ctrl, current_q_global
        try:
            k = key.char.lower()
        except AttributeError:
            k = None

        # Velocity control keys
        if k == 'w':
            vel_cmd.increase_vx()
            print(f"Forward: vx={vel_cmd.vx:.2f}")
        elif k == 's':
            vel_cmd.decrease_vx()
            print(f"Backward: vx={vel_cmd.vx:.2f}")
        elif k == 'a':
            vel_cmd.increase_vy()
            print(f"Left: vy={vel_cmd.vy:.2f}")
        elif k == 'd':
            vel_cmd.decrease_vy()
            print(f"Right: vy={vel_cmd.vy:.2f}")
        elif k == 'q':
            vel_cmd.increase_vyaw()
            print(f"CCW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'e':
            vel_cmd.decrease_vyaw()
            print(f"CW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'x':
            vel_cmd.stop_vx()
            print("Stop vx")
        elif k == 'c':
            vel_cmd.stop_vy()
            print("Stop vy")
        elif k == 'v':
            vel_cmd.stop_vyaw()
            print("Stop vyaw")

        # PD tuning hotkeys
        elif k == '[':
            if pd_tuner:
                pd_tuner.select_prev_type()
        elif k == ']':
            if pd_tuner:
                pd_tuner.select_next_type()
        elif k == 'p':
            if pd_tuner:
                pd_tuner.print_all_params()

        # Print preset pose
        elif k == 'm':
            pose_ctrl.print_preset_pose()

        # Adjust interpolation speed
        elif k == '+' or k == '=':
            pose_ctrl.increase_speed()
        elif k == '-' or k == '_':
            pose_ctrl.decrease_speed()

        # Special key controls
        if key == pynput_keyboard.Key.space:
            if current_q_global is not None:
                pose_ctrl.toggle_mode(current_q_global)
                vel_cmd.stop_all()
        elif key == pynput_keyboard.Key.up:
            if pd_tuner:
                pd_tuner.increase_kp()
        elif key == pynput_keyboard.Key.down:
            if pd_tuner:
                pd_tuner.decrease_kp()
        elif key == pynput_keyboard.Key.right:
            if pd_tuner:
                pd_tuner.increase_kd()
        elif key == pynput_keyboard.Key.left:
            if pd_tuner:
                pd_tuner.decrease_kd()
        elif key == pynput_keyboard.Key.page_up:
            if pd_tuner:
                pd_tuner.increase_ki()
        elif key == pynput_keyboard.Key.page_down:
            if pd_tuner:
                pd_tuner.decrease_ki()

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()


def get_foot_x_in_base(model, data):
    """
    计算两个 foot (轮子) 在基座坐标系中的 X 坐标

    Returns:
        tuple: (right_foot_x, left_foot_x) 相对于基座坐标系的 X 坐标 [m]
    """
    # 获取基座位姿
    base_pos = data.qpos[:3]  # 世界坐标系中基座位置
    base_quat = data.qpos[3:7]  # [w, x, y, z] MuJoCo 格式

    # 转换四元数格式用于 scipy (wxyz -> xyzw)
    base_quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
    r_base = R.from_quat(base_quat_scipy)

    # 获取 foot body 在世界坐标系中的位置
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")

    right_foot_pos_world = data.xpos[right_foot_id]
    left_foot_pos_world = data.xpos[left_foot_id]

    # 将 foot 位置从世界坐标系转换到基座坐标系
    # p_base = R_base^T * (p_world - base_pos)
    right_foot_in_base = r_base.apply(right_foot_pos_world - base_pos, inverse=True)
    left_foot_in_base = r_base.apply(left_foot_pos_world - base_pos, inverse=True)

    return right_foot_in_base[0], left_foot_in_base[0]


def get_obs(data, vel_cmd, last_action, debug=False):
    """
    Build observation vector (27D)

    Observation order (matches Isaac Lab training):
    - [0:3] base_ang_vel: base angular velocity (scaled 0.25)
    - [3:6] projected_gravity
    - [6:9] velocity_commands: [vx, vy, vyaw]
    - [9:15] joint_pos: joint positions (wheel positions = 0)
    - [15:21] joint_vel: joint velocities (scaled 0.05)
    - [21:27] actions: previous action
    """
    # Joint positions (relative to default pose)
    q = data.qpos[dof_ids].astype(np.double) - default_angle
    # Zero wheel positions (infinite rotation, position meaningless)
    q[-2:] = 0.0

    # Joint velocities
    dq = data.qvel[dof_vel].astype(np.double) * 0.05  # Scale

    # IMU data
    imu_quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)  # xyzw -> wxyz
    r_imu = R.from_quat(imu_quat)

    # Projected gravity
    proj = r_imu.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    # Angular velocity
    gyro_local = data.sensor('angular-velocity').data.astype(np.double)
    base_quat = data.qpos[3:7][[1, 2, 3, 0]].astype(np.double)
    r_base = R.from_quat(base_quat)
    gyro_world = r_imu.apply(gyro_local)
    gyro = r_base.apply(gyro_world, inverse=True) * 0.25  # Scale

    # Assemble observation vector
    obs = np.concatenate([
        gyro,      # [0:3] base_ang_vel
        proj,      # [3:6] projected_gravity
        np.array([vel_cmd.vx, vel_cmd.vy, vel_cmd.dyaw]),  # [6:9] velocity_commands
        q,         # [9:15] joint_pos
        dq,        # [15:21] joint_vel
        last_action  # [21:27] actions
    ]).astype(np.float32)

    if debug:
        print(f"gyro: {gyro}, proj: {proj}, q: {q}, dq: {dq}")

    # 返回 obs 和 proj 用于打印
    return obs, proj


def scale_action(raw_action, cfg):
    """
    Scale action

    Action order (6D):
    - [0:4] Leg joint positions (scale leg_scale)
    - [4:6] Wheel velocities (scale wheel_scale)
    """
    scaled = np.zeros_like(raw_action)
    # Leg joints
    scaled[:4] = raw_action[:4] * cfg.robot_config.leg_scale
    # Wheel velocities
    scaled[4:] = raw_action[4:] * cfg.robot_config.wheel_scale
    return scaled


def plot_joint_data(plot_data):
    """Plot joint tracking data"""
    joint_names = [
        "right_thigh", "right_calf",
        "left_thigh", "left_calf",
        "right_wheel", "left_wheel"
    ]

    time = np.array(plot_data['time'])

    # Figure 1: Target vs Actual
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
    fig1.suptitle('Helios Leg: Target vs Actual', fontsize=14, fontweight='bold')

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes1[row, col]

        targets = np.array(plot_data['targets'][i])
        actuals = np.array(plot_data['actuals'][i])

        ax.plot(time, targets, 'b--', linewidth=1.5, label='Target', alpha=0.8)
        ax.plot(time, actuals, 'g-', linewidth=1.5, label='Actual', alpha=0.8)

        if i < 4:
            ax.set_ylabel('Position [rad]', fontsize=9)
        else:
            ax.set_ylabel('Velocity [rad/s]', fontsize=9)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(joint_names[i], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout()
    fig1.savefig('helios_leg_tracking.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Figure saved: helios_leg_tracking.png")

    # Figure 2: Error and Torque
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle('Helios Leg: Error and Torque', fontsize=14, fontweight='bold')

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes2[row, col]

        errors = np.array(plot_data['errors'][i])
        torques = np.array(plot_data['torques'][i])

        ax.plot(time, errors, 'b-', linewidth=1.5, label='Error', alpha=0.8)
        if i < 4:
            ax.set_ylabel('Error [rad]', color='b', fontsize=9)
        else:
            ax.set_ylabel('Error [rad/s]', color='b', fontsize=9)
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(time, torques, 'r-', linewidth=1.5, label='Torque', alpha=0.8)
        ax2.set_ylabel('Torque [Nm]', color='r', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title(joint_names[i], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout()
    fig2.savefig('helios_leg_error_torque.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Figure saved: helios_leg_error_torque.png")

    plt.show()


def wait_for_manual_start(viewer):
    """在执行仿真前保持 MuJoCo 画面静止，方便检查初始姿态"""
    resume_event = threading.Event()

    def _wait_input():
        input("\n[PAUSE] MuJoCo 已暂停，按回车继续仿真...\n")
        resume_event.set()

    threading.Thread(target=_wait_input, daemon=True).start()
    print("\n[PAUSE] 画面已冻结，可观察机器人初始姿态，按终端回车恢复仿真。")
    while not resume_event.is_set():
        viewer.render()
        time.sleep(0.01)
    print("[PAUSE] 已收到输入，开始仿真。\n")


def run_mujoco(policy, mujoco_model_path, sim_duration, dt, decimation,
               debug=False, plot=False, keyboard_control=False, pause_on_start=False):
    """Run MuJoCo simulation"""
    global pd_tuner

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 启用坐标系显示 (frame = body frame)
    viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_BODY

    # Initialize PDTuner
    pd_tuner = PDTuner(cfg.robot_config)
    
   
        
    if keyboard_control:
        print("\n" + "="*70)
        print("Helios Leg Keyboard Controls")
        print("="*70)
        print("  Velocity Control:")
        print("    W/S: Forward/Backward (vx) - Note: may not work for this robot")
        print("    A/D: Left/Right (vy) - Primary movement direction")
        print("    Q/E: Counter-Clockwise/Clockwise (yaw)")
        print("")
        print("  Control Mode Toggle (Space - 4 stages):")
        print("    1st press: FREEZE - Stop target_q updates")
        print("    2nd press: INTERPOLATE to Pose 1")
        print("    3rd press: INTERPOLATE to Pose 2")
        print("    4th press: RESUME normal policy control")
        print("")
        print("  PID Tuning:")
        print("    [/]: Select joint type (thigh/calf/foot)")
        print("    Up/Down: Increase/Decrease Kp")
        print("    Left/Right: Decrease/Increase Kd")
        print("    PageUp/PageDown: Increase/Decrease Ki")
        print("    P: Print all PID parameters")
        print("")
        print("  Info:")
        print("    M: Show preset pose configurations")
        print("    +/-: Adjust interpolation speed")
        print("="*70 + "\n")

    # Set initial state
    data.qpos[:3] = [0, 0, cfg.robot_config.init_height]
    data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion [w, x, y, z]
    data.qpos[dof_ids] = default_angle.copy()
    data.qvel[:] = 0.0

    target_q = default_angle.copy()
    action = np.zeros(6, dtype=np.float32)
    last_action = np.zeros(6, dtype=np.float32)

    # PID integral term
    integral_error = np.zeros(6, dtype=np.double)

    # Plot data recording
    if plot:
        plot_data = {
            'time': [],
            'targets': [[] for _ in range(6)],
            'actuals': [[] for _ in range(6)],
            'errors': [[] for _ in range(6)],
            'torques': [[] for _ in range(6)]
        }

    if pause_on_start:
        mujoco.mj_step(model, data)
        viewer.render()
        wait_for_manual_start(viewer)

    steps = int(sim_duration / dt)
    try:
        for step in tqdm(range(steps), desc="Simulating...", disable=True):
            # Update global joint positions
            global current_q_global
            q = data.qpos[dof_ids]
            dq = data.qvel[dof_vel]
            current_q_global = q.copy()
            #print(data)
            # Update interpolation
            if pose_ctrl.mode == ControlMode.INTERPOLATING:
                interpolated_q = pose_ctrl.update_interpolation(dt)
                if interpolated_q is not None:
                    target_q[:4] = interpolated_q
                    target_q[4:] = 0.0  # Wheel velocity = 0

            # Policy control (at decimation rate)
            if step % decimation == 0:
                if pose_ctrl.mode == ControlMode.NORMAL:
                    # Build observation
                    obs, proj_gravity = get_obs(data, vel_cmd, last_action, debug=debug)

                    # 计算 foot 在基座坐标系中的 X 坐标
                    right_foot_x, left_foot_x = get_foot_x_in_base(model, data)

                    # 打印重力投影分量和 foot X 坐标
                    print(f"\r[Gravity] x={proj_gravity[0]:+.3f} y={proj_gravity[1]:+.3f} z={proj_gravity[2]:+.3f} | "
                          f"[Foot X] R={right_foot_x:+.3f} L={left_foot_x:+.3f}", end="")

                    obs_tensor = torch.from_numpy(obs).to(dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        raw_action = policy(obs_tensor).cpu().numpy().squeeze()
                    action[:] = raw_action
                    if debug:
                        print("raw_action:", action)
                    if step > 100:
                        scaled_action = scale_action(action, cfg)
                        target_q = scaled_action + default_angle
                    else:
                        target_q = default_angle.copy()
                    last_action = action.copy()

            # PID control
            kp = cfg.robot_config.kp_array
            kd = cfg.robot_config.kd_array
            ki = cfg.robot_config.ki_array
            integral_limit = cfg.robot_config.integral_limit_array
            tau_limit = cfg.robot_config.tau_limit_array

            # Position error (leg joints)
            position_error = target_q[:4] - q[:4]
            velocity_error = 0.0 - dq[:4]

            # Update integral error
            integral_error[:4] += position_error * dt
            integral_error[:4] = np.clip(integral_error[:4], -integral_limit[:4], integral_limit[:4])

            # Reset integral in interpolation or frozen mode
            if pose_ctrl.mode in [ControlMode.INTERPOLATING, ControlMode.FROZEN]:
                integral_error[:4] = 0.0

            tau = np.zeros(6)

            # Leg joints (0-3): PID position control
            tau[:4] = (kp[:4] * position_error +
                       ki[:4] * integral_error[:4] +
                       kd[:4] * velocity_error)

            # Wheel joints (4-5): velocity control
            tau[4:] = kd[4:] * (target_q[4:] - dq[4:])

            tau = np.clip(tau, -tau_limit, tau_limit)
            #tau = np.zeros(6)
            # Calculate errors (for plotting)
            errors = np.zeros(6)
            errors[:4] = target_q[:4] - q[:4]
            errors[4:] = target_q[4:] - dq[4:]

            # Record plot data
            if plot and step % decimation == 0:
                plot_data['time'].append(step * dt)
                for i in range(4):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(q[i])
                    plot_data['errors'][i].append(errors[i])
                    plot_data['torques'][i].append(tau[i])
                for i in range(4, 6):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(dq[i])
                    plot_data['errors'][i].append(errors[i])
                    plot_data['torques'][i].append(tau[i])
            # Apply torques
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            # 检测仿真不稳定
            if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
                print(f"\n[ERROR] 仿真不稳定! Time = {step * dt:.4f}s")
                print(f"  qpos: {data.qpos[:7]}")
                print(f"  qvel: {data.qvel[:6]}")
                print(f"  tau: {tau}")
                print(f"  target_q: {target_q}")
                break

            if step % decimation == 0:
                viewer.render()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        viewer.close()

        if plot and len(plot_data['time']) > 0:
            plot_joint_data(plot_data)

        if keyboard_control:
            print("\nFinal PID Parameters:")
            pd_tuner.print_all_params()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helios Leg MuJoCo Deployment')
    parser.add_argument('--model-path', type=str,
                        default='/home/liu/Desktop/robot_lab/source/robot_lab/data/Robots/helios_leg/mjcf/helios_leg.xml',
                        help='Path to MuJoCo XML model')
    parser.add_argument('--policy-path', type=str,
                        default='/home/liu/Desktop/robot_lab/logs/rsl_rl/helios_leg_flat/2025-12-08_23-13-58/exported/policy.pt',
                        help='Path to trained policy (.pt)')
    parser.add_argument('--duration', type=float, default=120.0, help='Simulation duration [s]')
    parser.add_argument('--dt', type=float, default=0.001, help='Physics timestep [s]')
    parser.add_argument('--decimation', type=int, default=5, help='Control decimation')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    parser.add_argument('--vx', type=float, default=0.0, help='Forward velocity command [m/s]')
    parser.add_argument('--vy', type=float, default=0.0, help='Lateral velocity command [m/s]')
    parser.add_argument('--vyaw', type=float, default=0.0, help='Yaw velocity command [rad/s]')
    parser.add_argument('--plot', action='store_true', help='Generate plots after simulation')
    parser.add_argument('--keyboard', action='store_true', help='Enable keyboard control')
    parser.add_argument('--pause-on-start', action='store_true', help='导入后先暂停渲染，等待手动继续')

    args = parser.parse_args()
    args.keyboard = True  # Enable keyboard control by default

    cfg.sim_config.dt = args.dt
    cfg.sim_config.decimation = args.decimation

    vel_cmd.vx = args.vx
    vel_cmd.vy = args.vy
    vel_cmd.dyaw = args.vyaw

    # Print PID parameters
    cfg.robot_config.print_parameters()

    # Print preset pose
    pose_ctrl.print_preset_pose()

    print(f"Loading policy from: {args.policy_path}")
    policy = torch.jit.load(args.policy_path)
    print(f"Velocity command: vx={vel_cmd.vx:.2f}, vy={vel_cmd.vy:.2f}, vyaw={vel_cmd.dyaw:.2f}")

    if args.keyboard:
        start_keyboard_listener()
        print("\n" + "="*70)
        print("Keyboard Control Enabled")
        print("="*70)
        print("  Velocity: W/S (vx), A/D (vy), Q/E (yaw)")
        print("  Mode: Space (toggle freeze/interpolate/resume)")
        print("  PID Tuning: [/] (select), Up/Down (Kp), Left/Right (Kd)")
        print("  Info: M (show poses), P (print PID)")
        print("="*70)

    if args.plot:
        print("\nPost-simulation plot enabled")

    print()
    run_mujoco(policy, args.model_path, args.duration, args.dt, args.decimation,
               debug=args.debug, plot=args.plot, keyboard_control=args.keyboard,
               pause_on_start=args.pause_on_start)
