# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog 轮腿机器人平坦地形环境配置

本文件包含：
1. MyDogFlatEnvCfg - 标准 PPO 版本（无历史观测）
2. MyDogHistFlatEnvCfg - HIM 版本（带 5 帧历史观测）
3. MyDogHandstandFlatEnvCfg - 倒立训练专用配置
"""

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import MyDogRoughEnvCfg, MyDogHistRoughEnvCfg


# ==============================================================================
# 标准 PPO 版本 - Flat 地形
# ==============================================================================


@configclass
class MyDogFlatEnvCfg(MyDogRoughEnvCfg):
    """标准 PPO 版本 - 平坦地形"""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 覆盖奖励
        self.rewards.base_height_l2.params["sensor_cfg"] = None
       
        # 强制平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 禁用高度扫描
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # 禁用课程
        self.curriculum.terrain_levels = None

        # 删除权重为0的奖励
        if self.__class__.__name__ == "MyDogFlatEnvCfg":
            self.disable_zero_weight_rewards()


# ==============================================================================
# HIM 版本 - Flat 地形
# ==============================================================================


@configclass
class MyDogHistFlatEnvCfg(MyDogHistRoughEnvCfg):
    """
    HIM 版本 - 平坦地形
    
    用于：
    1. 初期在平坦地形上预训练（更容易收敛）
    2. 测试和调试 HIM 框架
    
    与 MyDogHistRoughEnvCfg 的区别：
    1. 地形类型为平面
    2. 禁用高度扫描（height_scan_group）
    3. 禁用地形课程学习
    """

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # 覆盖高度相关奖励
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        
        # 强制平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 禁用高度扫描（Flat 环境不需要）
        self.scene.height_scanner = None
        # HIM 版本的 height_scan 在 height_scan_group 中，直接禁用整个组
        self.observations.height_scan_group = None
        
        # 禁用地形课程
        self.curriculum.terrain_levels = None

        # 删除权重为 0 的奖励
        if self.__class__.__name__ == "MyDogHistFlatEnvCfg":
            self.disable_zero_weight_rewards()


# ==============================================================================
# 倒立训练专用配置
# ==============================================================================


@configclass
class MyDogHandstandFlatEnvCfg(MyDogFlatEnvCfg):
    """专用于平地倒立训练的配置，启用手倒立奖励并关闭与行走相冲突的项。"""

    def __post_init__(self):
        super().__post_init__()

        # 仅在原地训练，关闭速度/朝向指令
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # 移除与行走相关的奖励，避免与倒立目标冲突
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = 0
        self.rewards.upward.weight = 0
        self.rewards.contact_forces.weight = 0

        # 放宽 root 相关惩罚，重点关注倒立姿态
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.base_height_l2.weight = 0

        # ------------------------------Handstand Rewards------------------------------
        handstand_type = "back"  # 使用前腿支撑倒立
        if handstand_type == "front":
            air_foot_pattern = "F.*_foot"      # 前轮悬空检测 + 脚高度检测
            knee_patterns = ["F.*(hip|thigh|calf)"]
            target_gravity = [-1.0, 0.0, 0.0]
        else:
            air_foot_pattern = "R.*_foot"      # 后轮悬空检测 + 脚高度检测
            knee_patterns = ["R.*(hip|thigh|calf)"]
            target_gravity = [1.0, 0.0, 0.0]

        self.rewards.handstand_orientation_l2.weight = -1.0
        self.rewards.handstand_orientation_l2.params["target_gravity"] = target_gravity

        # 脚底高度奖励 - 使用线性奖励，脚抬得越高奖励越大
        self.rewards.handstand_calf_height_linear.weight = 4.0
        self.rewards.handstand_calf_height_linear.params["asset_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_calf_height_linear.params["min_height"] = 0.8  # 最小高度 0.8m
        self.rewards.handstand_calf_height_linear.params["max_height"] = 1.0 # 最大高度 1.0m，超过则饱和

        # 关闭原来的指数型高度奖励
        self.rewards.handstand_feet_height_exp.weight = 0

        self.rewards.handstand_feet_on_air.weight = 1.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_feet_on_air.params["threshold"] = 5.0
        self.rewards.handstand_feet_on_air.params["knee_body_names"] = knee_patterns

        self.rewards.handstand_feet_air_time.weight = 1.0  # 提高权重
        self.rewards.handstand_feet_air_time.params["_sensor_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_feet_air_time.params["_threshold"] = 0.3  # 降低门槛，更容易获得正奖励
        self.rewards.handstand_feet_air_time.params["_knee_body_names"] = knee_patterns
        self.rewards.handstand_feet_air_time.params["_contact_force_threshold"] = 5.0

        # 前腿不良接触惩罚 - 只允许前轮接触地面，惩罚前腿 hip/thigh/calf 接触地面
        self.rewards.handstand_front_leg_undesired_contacts.weight = -5.0
        self.rewards.handstand_front_leg_undesired_contacts.params["sensor_cfg"].body_names = ["F.*(hip|thigh|calf)"]
        self.rewards.handstand_front_leg_undesired_contacts.params["threshold"] = 5.0

        # 身体接触地面惩罚 - 机器人躯干(base_link)接触地面时给予惩罚
        self.rewards.handstand_body_contact.weight = -5.0
        self.rewards.handstand_body_contact.params["threshold"] = 10.0

        # 倒立专用速度惩罚 - 保持静止
        self.rewards.handstand_lin_vel_xy_l2.weight = -1.5   # 惩罚 YZ 方向移动
        self.rewards.handstand_ang_vel_xyz_l2.weight = -1.5  # 惩罚旋转

        # ------------------------------Terminations------------------------------
        # 注意：bad_orientation 会在 |gravity_z| > threshold 时终止
        # 倒立目标是 gravity_z ≈ 0，但初始姿态是站立 (gravity_z = -1)
        # 如果开启，会导致刚开始就终止，无法学习
        # 建议：训练初期关闭，等学会倒立后再开启微调

        # 删除权重为0的奖励
        if self.__class__.__name__ == "MyDogHandstandFlatEnvCfg":
            self.disable_zero_weight_rewards()
