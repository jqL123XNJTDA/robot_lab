# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg (LW-360 Gen2V1) 双足轮腿机器人跳跃环境配置。

目标功能：
1. 跳跃前保持向前运动
2. 接收跳跃指令后向前跳跃
3. 达到目标高度（约 0.45-0.5m）
4. 稳定落地并继续向前运动

状态机流程：
    向前运动 (jump_cmd=0) → 蓄力 (0<jump_cmd<1) → 起跳 (jump_cmd=1) → 腾空 → 落地
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

    # 运动阶段（蓄力前）：保持目标高度惩罚
    jump_pre_charge_height = RewTerm(
        func=mdp.jump_pre_charge_height,
        weight=-50.0,  # 负权重，惩罚偏离目标高度
        params={
            "command_name": "jump_command",
            "target_height": 0.32,  # 运动时目标高度
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 运动阶段（蓄力前）：X 速度追踪奖励
    jump_pre_charge_vel_tracking = RewTerm(
        func=mdp.jump_pre_charge_vel_tracking,
        weight=6.0,
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 蓄力阶段：压低重心奖励
    jump_charge_crouch = RewTerm(
        func=mdp.jump_charge_crouch,
        weight=10.0,
        params={
            "command_name": "jump_command",
            "target_height": 0.25,  # 蓄力时目标高度（压低）
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 蓄力阶段：X 速度追踪奖励
    jump_charge_vel_tracking = RewTerm(
        func=mdp.jump_charge_vel_tracking,
        weight=4.0,  # 蓄力时速度追踪权重稍低
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 蓄力阶段：双脚着地惩罚（离地惩罚）
    jump_charge_feet_contact = RewTerm(
        func=mdp.jump_charge_feet_contact,
        weight=-30.0,  # 负权重，惩罚离地
        params={
            "command_name": "jump_command",
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_link"]),
        },
    )

    # 起跳阶段：向上速度奖励
    jump_launch_upward = RewTerm(
        func=mdp.jump_launch_upward,
        weight=0.0,
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 起跳阶段：垂直爆发速度奖励（速度越大，奖励越高）
    jump_launch_vertical_velocity = RewTerm(
        func=mdp.jump_launch_vertical_velocity,
        weight=50.0,
        params={
            "command_name": "jump_command",
            "velocity_scale": 1.0,  # 速度缩放系数
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 起跳阶段：地面反作用力奖励（蹬地力度越大，奖励越高）
    jump_launch_grf = RewTerm(
        func=mdp.jump_launch_grf,
        weight=50.0,
        params={
            "command_name": "jump_command",
            "max_force": 700.0,  # 最大力值上限 [N]，防止物理不稳定
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_link"]),
        },
    )

    # 腾空阶段：高度奖励
    jump_flight_height = RewTerm(
        func=mdp.jump_flight_height,
        weight=0.0,
        params={
            "command_name": "jump_command",
            "target_height": 0.5,  # 目标跳跃高度
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

    # 腾空阶段：动作变化率惩罚（禁止空中动作剧烈变化）
    jump_flight_action_rate_penalty = RewTerm(
        func=mdp.jump_flight_action_rate_penalty,
        weight=0.0,  # 负权重惩罚
        params={
            "command_name": "jump_command",
        },
    )

    # 落地阶段：稳定性奖励
    jump_land_stable = RewTerm(
        func=mdp.jump_land_stable,
        weight=0,
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
        weight=0,  # 负权重，惩罚 Z 轴速度
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 落地阶段：X 速度追踪奖励（恢复前向运动）
    jump_land_vel_tracking = RewTerm(
        func=mdp.jump_land_vel_tracking,
        weight=0,
        params={
            "command_name": "jump_command",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 全阶段角速度追踪（非腾空阶段） ===
    # Z 轴角速度追踪奖励（使用 jump_command 替代 base_velocity）
    jump_track_ang_vel_z = RewTerm(
        func=mdp.jump_track_ang_vel_z,
        weight=3.0,  # 与原 track_ang_vel_z_exp 权重相近
        params={
            "command_name": "jump_command",
            "std": 0.25,  # 指数核标准差
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 基础约束奖励（跳跃时仍需要） ===

    # 终止惩罚
    jump_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # 平坦姿态惩罚（蓄力和落地时需要）
    jump_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 动作平滑惩罚
    jump_action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)

    # 关节力矩惩罚（节能）
    jump_joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # 左右对称惩罚
    jump_joint_mirror = RewTerm(
        func=mdp.joint_mirror_neg,
        weight=-0.0,
        params={
            "mirror_joints": [
                ["right_thigh_joint", "left_thigh_joint"],
                ["right_calf_joint", "left_calf_joint"],
            ],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === 错误时机跳跃惩罚 ===
    # 当 jump_cmd < 0.5 时惩罚向上速度，防止不该跳时乱跳
    jump_wrong_timing_penalty = RewTerm(
        func=mdp.jump_wrong_timing_penalty,
        weight=-0.0,  # 负权重惩罚
        params={
            "command_name": "jump_command",
            "jump_cmd_threshold": 0.8,  # jump_cmd < 0.8 时惩罚
            "asset_cfg": SceneEntityCfg("robot"),
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

        
        self.rewards.flat_orientation_l2.weight = -50
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.is_terminated.weight = -100

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
        self.rewards.joint_mirror_neg.weight = -20
        self.rewards.joint_mirror_neg.params["mirror_joints"] = [
            ["right_thigh_joint", "left_thigh_joint"],
            ["right_calf_joint", "left_calf_joint"],
        ]

        # === 动作惩罚 ===
        # 动作变化率惩罚（平滑动作）
        self.rewards.action_rate_l2.weight = 0.0

        # === 接触传感器相关 ===
        # 非期望接触惩罚（除轮子外的接触）
        self.rewards.undesired_contacts.weight = -5.0
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
            charge_duration=0.5,  # 蓄力时间 0.5s
            resampling_time_range=(6.0, 6.0),  # episode 重采样时间
            debug_vis=False,
            ranges=mdp.JumpCommandCfg.Ranges(
                lin_vel_x=(0.5, 1.0),  # 跳跃前向前运动速度
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
        
        # ------------------------------Episode 配置------------------------------
        self.episode_length_s = 6.0
        
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
