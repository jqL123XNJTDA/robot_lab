# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg (LW-360 Gen2V1) 双足轮腿机器人跳跃环境配置。

目标功能：
1. 垂直跳跃（原地起跳）
2. 达到目标高度（约 0.45-0.5m）
3. 稳定落地

简化状态机流程（参考 GO2_Spring_Jump）：
    待机 (jump_cmd=0) → 跳跃触发 (jump_cmd=1) → 腾空 (was_in_flight) → 落地 (has_jumped)
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import HeliosLegFlatEnvCfg
from .rough_env_cfg import HeliosLegRewardsCfg


@configclass
class HeliosLegJumpRewardsCfg(HeliosLegRewardsCfg):
    """Helios 跳跃奖励配置

    继承 HeliosLegRewardsCfg，在 __post_init__ 中清零所有继承的奖励，只保留跳跃专用奖励。
    """

    # === 清零所有继承的奖励（在 __post_init__ 中处理） ===

    # === 跳跃分阶段奖励 ===

    # 待机阶段（jump_cmd=0）：保持目标高度惩罚
    jump_pre_charge_height = RewTerm(
        func=mdp.jump_pre_charge_height,
        weight=-100,  # 负权重，惩罚偏离目标高度
        params={
            "command_name": "jump_command",
            "target_height": 0.25,  # 待机时目标高度
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

  

 
    # 腾空阶段：高度奖励
    jump_flight_height = RewTerm(
        func=mdp.jump_flight_height,
        weight=100.0,
        params={
            "command_name": "jump_command",
            "target_height": 0.5,  # 目标跳跃高度
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 腾空阶段：Z 轴速度奖励（参考 GO2 _reward_line_z）
    jump_flight_vel_z = RewTerm(
        func=mdp.jump_flight_vel_z,
        weight=100.0,  # 参考 GO2 权重
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 腾空阶段：动作惩罚（禁止空中乱蹬腿，防止角动量守恒导致翻车）
    jump_flight_action_penalty = RewTerm(
        func=mdp.jump_flight_action_penalty,
        weight=0.0,  # 负权重惩罚
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"]),
        },
    )

    # 腾空阶段：脚部高度惩罚（防止空中乱蹬腿，目标是脚在 base 下方 0.2m）
    jump_flight_foot_height = RewTerm(
        func=mdp.jump_flight_foot_height_penalty,
        weight=-100.0,  # 负权重惩罚
        params={
            "command_name": "jump_command",
            "target_height_below": 0.1,  # 脚应在 base 下方的目标距离 [m]
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": ["left_foot_link", "right_foot_link"],
        },
    )

    # 落地阶段：稳定性奖励
    jump_land_stable = RewTerm(
        func=mdp.jump_land_stable,
        weight=0.5,
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 落地阶段：恢复目标高度惩罚
    jump_land_height = RewTerm(
        func=mdp.jump_land_height,
        weight=0,  # 负权重，惩罚偏离目标高度
        params={
            "command_name": "jump_command",
            "target_height": 0.32,  # 落地后恢复到运动时目标高度
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 落地阶段：垂直速度阻尼惩罚（软着陆，消除垂直震荡）
    jump_land_vertical_velocity = RewTerm(
        func=mdp.jump_land_vertical_velocity,
        weight=-2,  # 负权重，惩罚 Z 轴速度
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # === 全阶段角速度追踪（原地跳跃禁用） ===
    jump_track_ang_vel_z = RewTerm(
        func=mdp.jump_track_ang_vel_z,
        weight=1,  # 原地跳跃，禁用角速度追踪
        params={
            "command_name": "jump_command",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 全阶段 X 速度追踪 ===
    jump_track_lin_vel_x = RewTerm(
        func=mdp.jump_track_lin_vel_x,
        weight=0.5,  # 全阶段跟踪前向速度
        params={
            "command_name": "jump_command",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 基础约束奖励（跳跃时仍需要） ===


  

    # 动作平滑惩罚
    jump_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # === 错误时机跳跃惩罚 ===
    # 当 jump_cmd < 0.5 时惩罚向上速度，防止不该跳时乱跳
  
    # === 待机和落地阶段双脚离地惩罚 ===
    # 在 jump_cmd=0（待机）和 has_jumped=True（落地后）阶段，惩罚双脚离地
    jump_idle_land_feet_air = RewTerm(
        func=mdp.jump_idle_land_feet_air_penalty,
        weight=-5.0,  # 负权重惩罚
        params={
            "command_name": "jump_command",
            "contact_threshold": 1.0,  # 接触力阈值 [N]
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_link"]),
        },
    )


@configclass
class HeliosLegJumpEnvCfg(HeliosLegFlatEnvCfg):
    """Helios 双足轮腿跳跃环境配置

    继承自 FlatEnvCfg，使用全新的跳跃奖励配置。
    """

    rewards: HeliosLegJumpRewardsCfg = HeliosLegJumpRewardsCfg()

    def __post_init__(self):
        # 调用父类的 __post_init__
        super().__post_init__()

        # ------------------------------清零所有继承的奖励------------------------------
        # 遍历 rewards 的所有属性，把非跳跃专用的奖励权重设为 0
        for attr_name in dir(self.rewards):
            if attr_name.startswith("_") or attr_name.startswith("jump_"):
                continue
            attr = getattr(self.rewards, attr_name, None)
            if isinstance(attr, RewTerm):
                attr.weight = 0.0

        
        self.rewards.flat_orientation_l2.weight = -120
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.is_terminated.weight = -300

        # 基座高度惩罚（禁用，改用 base_height_reward）
        # self.rewards.base_height_l2.weight = -50
        # self.rewards.base_height_l2.params["target_height"] = 0.32
        # self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 基座线加速度惩罚（平滑运动）
        self.rewards.body_lin_acc_l2.weight = -1e-6
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

 
        self.rewards.joint_pos_limits.weight = -5
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
  
   
        # 反向镜像奖励（左右关节角度符号相反: left = -right）
        self.rewards.joint_mirror_neg.weight = -30
        self.rewards.joint_mirror_neg.params["mirror_joints"] = [
            ["right_thigh_joint", "left_thigh_joint"],
            ["right_calf_joint", "left_calf_joint"],
        ]

        # === 动作惩罚 ===
        # 动作变化率惩罚（平滑动作）
        self.rewards.action_rate_l2.weight = -0.02

        # === 接触传感器相关 ===
        # 非期望接触惩罚（除轮子外的接触）
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0


        # 禁用速度跟踪奖励（依赖 base_velocity 命令，跳跃环境使用 jump_command）
        # track_lin_vel_x_exp 和 track_ang_vel_z_exp 使用 base_velocity 命令，需要禁用
        self.rewards.track_lin_vel_x_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

      
        # ------------------------------Commands 命令配置------------------------------
        # 添加跳跃命令（替换基础速度命令）
        self.commands.jump_command = mdp.JumpCommandCfg(
            asset_name="robot",
            contact_sensor_name="contact_forces",
            feet_body_names=[".*_foot_link"],
            contact_threshold=1.0,
            jump_trigger_range=(50, 100),  # 50-100 帧后触发跳跃
            resampling_time_range=(6.0, 6.0),  # episode 重采样时间
            debug_vis=False,
            ranges=mdp.JumpCommandCfg.Ranges(
                lin_vel_x=(0, 0.0),  # 原地跳跃，无前向速度
                ang_vel_z=(0.0, 0.0),  # 无旋转
            ),
        )

        # 禁用基础速度命令
        self.commands.base_velocity = None

        # ------------------------------Observations 观测配置------------------------------
        # 修改速度命令观测，使用跳跃命令
        self.observations.policy.velocity_commands.params["command_name"] = "jump_command"
        self.observations.critic.velocity_commands.params["command_name"] = "jump_command"

        # ------------------------------Events 事件配置------------------------------
        # 添加跳跃辅助推力事件（训练初期帮助学习跳跃）
        self.events.push_upward_for_jump = EventTerm(
            func=mdp.push_robot_upward_for_jump,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # 每步检查
            params={
                "command_name": "jump_command",
                "velocity_range": (1.5, 2.5),  # 向上速度范围 [m/s]
                "probability": 0.8,  # 训练初期 80% 概率给辅助推力
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.events.push_upward_for_jump = None
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        # ------------------------------Episode 配置------------------------------
        self.episode_length_s = 5.0
        
        # ------------------------------Curriculum 课程学习配置------------------------------
        # 禁用速度课程（跳跃不需要）
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # 自动移除权重为0的奖励项
        if self.__class__.__name__ == "HeliosLegJumpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class HeliosLegJumpEnvCfg_NoAssist(HeliosLegJumpEnvCfg):
    """无辅助推力的跳跃环境配置（用于后期训练）"""

    def __post_init__(self):
        super().__post_init__()

        # 禁用辅助推力
        self.events.push_upward_for_jump = None

        if self.__class__.__name__ == "HeliosLegJumpEnvCfg_NoAssist":
            self.disable_zero_weight_rewards()


@configclass
class HeliosLegJumpEnvCfg_LowAssist(HeliosLegJumpEnvCfg):
    """低辅助推力的跳跃环境配置（用于中期训练）"""

    def __post_init__(self):
        super().__post_init__()

        # 降低辅助推力概率
        self.events.push_upward_for_jump.params["probability"] = 0.3

        if self.__class__.__name__ == "HeliosLegJumpEnvCfg_LowAssist":
            self.disable_zero_weight_rewards()
