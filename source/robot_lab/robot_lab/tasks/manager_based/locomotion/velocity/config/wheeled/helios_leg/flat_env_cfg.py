# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg (LW-360 Gen2V1) 双足轮腿机器人 Flat 环境配置。
继承自 RoughEnvCfg，使用平坦地形进行初始策略学习。

提供两个版本：
- HeliosLegFlatEnvCfg: 标准 PPO 配置
- HeliosLegHistFlatEnvCfg: HIM 风格配置（带5帧历史观测）
"""

from isaaclab.utils import configclass

from .rough_env_cfg import HeliosLegRoughEnvCfg, HeliosLegHistRoughEnvCfg


@configclass
class HeliosLegFlatEnvCfg(HeliosLegRoughEnvCfg):
    """标准 PPO Flat 环境配置"""

    def __post_init__(self):
        # post init of parent class
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # 强制平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 禁用高度扫描器
        self.scene.height_scanner = None
        # 禁用 height_scan 观测（标准 PPO 使用 height_scan）
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # 禁用课程
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HeliosLegFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class HeliosLegHistFlatEnvCfg(HeliosLegHistRoughEnvCfg):
    """HIM 风格 Flat 环境配置 - 带5帧历史观测

    基于 HIM (Hybrid Internal Model) 论文：
    - Policy 只用本体感知（不含 base_lin_vel）
    - Critic 可访问特权信息（含 base_lin_vel）
    - 5帧历史观测用于提取环境动态信息
    - Flat 环境禁用 height_scan_group
    """

    def __post_init__(self):
        # post init of parent class
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # 强制平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 禁用高度扫描器
        self.scene.height_scanner = None
        # 禁用 height_scan_group 观测组（HIM 风格：Critic 高度扫描单独分组）
        self.observations.height_scan_group = None

        # 禁用课程
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HeliosLegHistFlatEnvCfg":
            self.disable_zero_weight_rewards()
