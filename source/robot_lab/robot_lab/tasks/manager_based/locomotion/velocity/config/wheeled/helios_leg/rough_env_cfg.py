# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg (LW-360 Gen2V1) 双足轮腿机器人 Rough 环境配置。

机器人结构：
- 4个腿部关节（位置控制）：right_thigh, right_calf, left_thigh, left_calf
- 2个轮子关节（速度控制）：right_foot, left_foot
- 总计6个自由度
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.helios_leg import HELIOS_LEG_CFG  # isort: skip

# 地形配置
ROUGH_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class HeliosLegActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    # 腿部关节位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    # 轮子关节速度控制
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class HeliosLegRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # 轮子速度惩罚
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # 轮子加速度惩罚
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # 轮子力矩惩罚
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # Y方向线速度跟踪奖励（只跟踪侧向移动，适用于双轮腿机器人）
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=0.0,
        params={"std": 0.5, "command_name": "base_velocity"},
    )

    # X方向线速度跟踪奖励（只跟踪前后移动，适用于无髋关节的双轮腿机器人）
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=0.0,
        params={"std": 0.5, "command_name": "base_velocity"},
    )

    # 左右 foot X 方向对齐惩罚（使左右轮并列）
    feet_x_alignment_l2 = RewTerm(
        func=mdp.feet_x_alignment_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
    )

    # foot X 方向偏移惩罚（使轮子保持在目标 X 位置）
    feet_x_offset_l2 = RewTerm(
        func=mdp.feet_x_offset_l2,
        weight=0.0,
        params={"target_x": 0.0, "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
    )

    # 基座高度奖励（高度越高奖励越大）
    base_height_reward = RewTerm(
        func=mdp.base_height_reward,
        weight=0.0,
        params={"min_height": 0.0, "max_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class HeliosLegRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: HeliosLegActionsCfg = HeliosLegActionsCfg()
    rewards: HeliosLegRewardsCfg = HeliosLegRewardsCfg()

    # 机器人link名称
    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"  # 轮子link

    # fmt: off
    # 腿部关节（位置控制）：大腿和小腿，共4个
    leg_joint_names = [
        "right_thigh_joint", "right_calf_joint",
        "left_thigh_joint", "left_calf_joint",
    ]
    # 轮子关节（速度控制）：2个
    wheel_joint_names = [
        "right_foot_joint", "left_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # 调用父类的 __post_init__
        super().__post_init__()

        # ------------------------------Scene 场景配置------------------------------
        # 设置机器人资产配置
        self.scene.robot = HELIOS_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 高度扫描器挂载到 base_link
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # 地形类型设为生成器模式
        self.scene.terrain.terrain_type = "generator"
        # 使用粗糙地形配置
        self.scene.terrain.terrain_generator = ROUGH_ROAD_CFG

        # ------------------------------Observations 观测配置------------------------------
        # 排除轮子位置观测（轮子是continuous关节，无限旋转，位置无意义）
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        # 基座线速度观测缩放
        self.observations.policy.base_lin_vel.scale = 2.0
        # 基座角速度观测缩放
        self.observations.policy.base_ang_vel.scale = 0.25
        # 关节位置观测缩放
        self.observations.policy.joint_pos.scale = 1.0
        # 关节速度观测缩放
        self.observations.policy.joint_vel.scale = 0.05
        # 禁用基座线速度观测（轮腿机器人通常不需要）
        self.observations.policy.base_lin_vel = None
        # 禁用高度扫描观测
        self.observations.policy.height_scan = None
        # 设置观测的关节名称列表
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions 动作配置------------------------------
        # 关节位置动作缩放（不同关节不同缩放系数）
        self.actions.joint_pos.scale = {
            ".*_thigh_joint": 0.125,  # 大腿关节缩放较小，动作更精细
            ".*_calf_joint": 0.25,    # 小腿关节缩放
        }
        # 轮子速度动作缩放
        self.actions.joint_vel.scale = 5.0
        # 位置动作裁剪范围
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # 速度动作裁剪范围
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        # 位置控制的关节列表（腿部关节）
        self.actions.joint_pos.joint_names = self.leg_joint_names
        # 速度控制的关节列表（轮子关节）
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events 事件/领域随机化配置------------------------------
        # 基座刚体质量随机化
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 其他刚体质量随机化（排除base_link）
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 质心位置随机化
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 外力/力矩扰动应用到base_link
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # 禁用部分领域随机化（训练初期简化）
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_motor_overheat = None
        self.events.randomize_sensor_noise = None
        self.events.randomize_time_delay = None

        # 保留关节重置事件，但不加随机（使用 init_state 定义的默认姿态）
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)

        # 保留基座重置事件，无位置/速度随机化
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        # ------------------------------Rewards 奖励配置------------------------------
        # === 通用奖励 ===
        # 终止惩罚（摔倒等）
        self.rewards.is_terminated.weight = -200

        # === 基座/根部惩罚 ===
        # Z方向线速度惩罚（抑制上下抖动）- 双轮腿需要较强抑制
        self.rewards.lin_vel_z_l2.weight = -2
        # XY方向角速度惩罚（抑制翻滚/俯仰晃动）- 双轮腿容易翻滚，需加强
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 平坦姿态惩罚（鼓励保持水平）- 双轮腿平衡难度大，需加强
        self.rewards.flat_orientation_l2.weight = -10
        # 基座高度惩罚（禁用，改用 base_height_reward）
        self.rewards.base_height_l2.weight = 0
        # 基座高度奖励（高度越高奖励越大）- 正权重
        self.rewards.base_height_reward.weight = 2.0
        self.rewards.base_height_reward.params["min_height"] = 0.35  # 目标高度 0.35m
        self.rewards.base_height_reward.params["max_height"] = 0.4  # 最高高度 0.4m（饱和）
        # 基座线加速度惩罚（平滑运动）
        self.rewards.body_lin_acc_l2.weight = -1e-4
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # === 关节惩罚 ===
        # 腿部关节力矩惩罚（节能）
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 轮子关节力矩惩罚
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 腿部关节速度惩罚
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 轮子关节速度惩罚
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 腿部关节加速度惩罚（平滑运动）
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 轮子关节加速度惩罚
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 关节位置限位惩罚（避免触碰关节极限）
        self.rewards.joint_pos_limits.weight = -5
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        # 关节速度限位惩罚
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 关节功率惩罚（节能）
        self.rewards.joint_power.weight = 0
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        # 静止时关节运动惩罚（命令为零时保持静止）- 启用以增强静止稳定性
        self.rewards.stand_still.weight = -0.01
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        # 关节位置偏差惩罚
        self.rewards.joint_pos_penalty.weight = -1
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        #self.rewards.joint_pos_penalty.params["velocity_threshold"] = 100
        # 轮子速度与地面速度不匹配惩罚（防止打滑）
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 关节镜像奖励（鼓励左右对称运动）- 禁用，因为 helios_leg 左右关节是反向对称
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["right_(thigh|calf)_joint", "left_(thigh|calf)_joint"],
        ]
        # 反向镜像奖励（左右关节角度符号相反: left = -right）
        self.rewards.joint_mirror_neg.weight = -2
        self.rewards.joint_mirror_neg.params["mirror_joints"] = [
            ["right_thigh_joint", "left_thigh_joint"],
            ["right_calf_joint", "left_calf_joint"],
        ]

        # === 动作惩罚 ===
        # 动作变化率惩罚（平滑动作）
        self.rewards.action_rate_l2.weight = -0.01

        # === 接触传感器相关 ===
        # 非期望接触惩罚（除轮子外的接触）
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        # 接触力惩罚（轮子接触力过大）
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # === 速度跟踪奖励 ===
        # 禁用XY方向线速度跟踪
        self.rewards.track_lin_vel_xy_exp.weight = 0
        # 禁用Y方向线速度跟踪
        self.rewards.track_lin_vel_y_exp.weight = 0
        # 禁用X方向线速度跟踪（先专注平衡）
        self.rewards.track_lin_vel_x_exp.weight = 0
        # 禁用Z方向角速度跟踪（先专注平衡）
        self.rewards.track_ang_vel_z_exp.weight = 0

        # === 其他奖励 ===
        # 足部腾空时间奖励（对轮腿机器人通常禁用）
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足部接触奖励
        self.rewards.feet_contact.weight = 0.5
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 无命令时足部接触奖励（静止时保持接地）
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足部绊倒惩罚
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足部滑动惩罚
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 足部高度奖励
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # 足部相对身体高度奖励
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 步态奖励（左右轮同步）
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("left_foot_joint", "right_foot_joint"),)
        # 保持直立奖励（正权重，鼓励正立姿态）
        self.rewards.upward.weight = 1

        # === Foot 位置奖励 ===
        # 左右 foot X 方向对齐惩罚（使左右轮在 X 方向上并列）
        self.rewards.feet_x_alignment_l2.weight = -2.0
        self.rewards.feet_x_alignment_l2.params["asset_cfg"].body_names = [self.foot_link_name]
        # foot X 方向偏移惩罚（使轮子保持在目标 X 位置）
        # target_x=0 表示轮子应在基座正下方；根据 URDF 几何可调整为其他值
        self.rewards.feet_x_offset_l2.weight = -2.0
        self.rewards.feet_x_offset_l2.params["target_x"] = 0.0
        self.rewards.feet_x_offset_l2.params["asset_cfg"].body_names = [self.foot_link_name]
        

        # 自动移除权重为0的奖励项（优化性能）
        if self.__class__.__name__ == "HeliosLegRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件配置------------------------------
        # 非法接触终止：base_link 和 calf_link 接触地面时终止
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums 课程学习配置------------------------------
        # 禁用线速度命令课程（先专注平衡）
        self.curriculum.command_levels_lin_vel = None
        # 禁用角速度命令课程（先专注平衡）
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands 命令配置------------------------------
        # X方向线速度命令范围 (m/s) - 禁用，先专注平衡
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # Y方向线速度命令范围 (m/s) - 禁用
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # Z方向角速度命令范围 (rad/s) - 禁用，先专注平衡
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
