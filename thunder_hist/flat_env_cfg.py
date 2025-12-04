# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_hist.rough_env_cfg import (  # noqa: E501
    ThunderHistRoughEnvCfg,
)


@configclass
class ThunderHistFlatEnvCfg(ThunderHistRoughEnvCfg):
    """Thunder轮式机器人平坦地形配置 - 带5帧历史观测"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ThunderHistFlatEnvCfg":
            self.disable_zero_weight_rewards()
