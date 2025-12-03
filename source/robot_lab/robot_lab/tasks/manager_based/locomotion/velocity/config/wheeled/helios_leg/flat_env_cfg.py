# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg (LW-360 Gen2V1) 双足轮腿机器人 Flat 环境配置。
继承自 RoughEnvCfg，使用平坦地形进行初始策略学习。
"""

from isaaclab.utils import configclass

from .rough_env_cfg import HeliosLegRoughEnvCfg


@configclass
class HeliosLegFlatEnvCfg(HeliosLegRoughEnvCfg):
    def __post_init__(self):
        # post init of parent class
        super().__post_init__()

        # override rewards
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

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HeliosLegFlatEnvCfg":
            self.disable_zero_weight_rewards()
