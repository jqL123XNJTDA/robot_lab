# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog 轮腿机器人环境配置

本文件包含两个版本：
1. MyDogRoughEnvCfg - 标准 PPO 版本（无历史观测）
2. MyDogHistRoughEnvCfg - HIM 版本（带 5 帧历史观测，用于 HIM 训练）
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ObservationsCfg,
)

# 引入你的机器狗资产
from robot_lab.assets.mydog import MYDOG_CFG


# ==============================================================================
# 标准 PPO 版本 - 无历史观测
# ==============================================================================


@configclass
class MyDogActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    # 腿部关节：位置控制 (Hip, Thigh, Calf)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"], 
        scale=0.25, 
        use_default_offset=True, 
        clip=None, 
        preserve_order=True
    )
    # 轮子关节：速度控制 (Foot) - 对应 Go2W 的逻辑
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=[".*_foot_joint"], 
        scale=5.0, # 轮子速度缩放，可根据需要调整
        use_default_offset=True, 
        clip=None, 
        preserve_order=True
    )


@configclass
class MyDogRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    
    # --- 轮足机器人特有奖励 ---
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")}
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")}
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")}
    )

    # Handstand-specific shapers (默认权重为0，只有在倒立场景下启用)
    handstand_feet_height_exp = RewTerm(
        func=mdp.handstand_feet_height_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[ ".*_foot" ]),
            "target_height": 0.5,
            "std": math.sqrt(0.25),
        },
    )
    handstand_feet_on_air = RewTerm(
        func=mdp.handstand_feet_on_air,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_foot" ]),
            "threshold": 10.0,
            "knee_body_names": [".*(thigh|calf).*"],
        },
    )
    handstand_feet_air_time = RewTerm(
        func=mdp.HandstandFeetAirTimeReward,
        weight=0.0,
        params={
            "_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_foot" ]),
            "_asset_cfg": SceneEntityCfg("robot"),
            "_threshold": 0.4,
            "_knee_body_names": [".*(thigh|calf).*"],
            "_contact_force_threshold": 5.0,
        },
    )
    handstand_orientation_l2 = RewTerm(
        func=mdp.handstand_orientation_l2,
        weight=0.0,
        params={
            "target_gravity": [1.0, 0.0, 0.0],
        },
    )

    # 后腿脚底高度奖励（线性） - 脚越高奖励越大
    handstand_calf_height_linear = RewTerm(
        func=mdp.handstand_calf_height_linear,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["R.*_foot"]),
            "min_height": 0.6,  # 最小高度（m）
            "max_height": 1.0,  # 最大高度（m），超过此高度奖励饱和
        },
    )

    # 前腿不良接触惩罚
    handstand_front_leg_undesired_contacts = RewTerm(
        func=mdp.handstand_undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F.*(hip|thigh|calf)"]),
            "threshold": 1.0,
        },
    )

    # 身体接触地面惩罚
    handstand_body_contact = RewTerm(
        func=mdp.handstand_undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )

    # 倒立专用速度惩罚 - 保持静止
    handstand_lin_vel_xy_l2 = RewTerm(
        func=mdp.handstand_lin_vel_xy_l2,
        weight=0.0,
        params={},
    )

    handstand_ang_vel_xyz_l2 = RewTerm(
        func=mdp.handstand_ang_vel_xyz_l2,
        weight=0.0,
        params={},
    )


@configclass
class MyDogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """标准 PPO 版本 - 无历史观测"""
    
    actions: MyDogActionsCfg = MyDogActionsCfg()
    rewards: MyDogRewardsCfg = MyDogRewardsCfg()

    # [关键修改] 根据你的 URDF 设置 Link 名称
    base_link_name = "base_link" 
    foot_link_name = ".*_foot"

    # 定义关节组
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    # 合并列表
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # 替换机器人资产
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # 策略网络观察：使用不包含轮子角度的相对位置函数 (因为轮子角度是无限增加的)
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        
        # 观察值缩放
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None # 这里的 None 意味着使用默认配置或被后续逻辑覆盖
        self.observations.policy.height_scan = None
        
        # 确保观察包含所有关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # 动作缩放：腿部动作为位置，轮子动作为速度
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0 # 轮子速度增益
        
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        
        # 绑定动作对应的关节
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14), # 允许全向翻滚复位测试
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = 0

        # 姿态惩罚
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0 # 平地训练可以开启，Rouge地形可以设为0
        self.rewards.base_height_l2.weight = 0 # 可以在 Flat 配置中开启
        self.rewards.base_height_l2.params["target_height"] = 0.45 # 根据你的狗调整高度
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 关节惩罚 (区分腿和轮)
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        
        # 站立惩罚 (无指令时应该站住)
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        
        # 轮子速度惩罚 (在空中时)
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names

        # 动作变化率惩罚
        self.rewards.action_rate_l2.weight = -0.01

        # 接触惩罚
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度追踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # 其他奖励
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        # 防止"溜冰"：没有指令时，脚(轮子)不应该乱动
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        self.rewards.upward.weight = 1.0

        # 删除权重为0的奖励
        if self.__class__.__name__ == "MyDogRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None # 初始训练可以先放宽碰撞检测
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None


# ==============================================================================
# HIM 版本 - 带 5 帧历史观测 (用于 HIM 训练框架)
# ==============================================================================


@configclass
class MyDogHistCommandParams:
    """HIM 版本的命令参数"""
    # 速度命令范围
    lin_vel_x: tuple = (0.0,0.0)
    lin_vel_y: tuple = (0.0,0.0)
    ang_vel_z: tuple = (0, 0.0)


@configclass
class MyDogHistEventParams:
    """HIM 版本的事件随机化参数"""
    # 复位基座随机化
    reset_base_pose_range_x: tuple = (0, 0)
    reset_base_pose_range_y: tuple = (0, 0)
    reset_base_pose_range_z: tuple = (0, 0.2)
    reset_base_pose_range_roll: tuple = (0, 0)
    reset_base_pose_range_pitch: tuple = (0, 0)
    reset_base_pose_range_yaw: tuple = (0, 0)

    reset_base_velocity_range_x: tuple = (0, 0.5)
    reset_base_velocity_range_y: tuple = (0, 0)
    reset_base_velocity_range_z: tuple = (0, 0)
    reset_base_velocity_range_roll: tuple = (0, 0)
    reset_base_velocity_range_pitch: tuple = (0, 0)
    reset_base_velocity_range_yaw: tuple = (0, 0)

    # 外力/力矩随机化 - 与父类一致，初期训练不宜过大
    external_force_range: tuple = (-10.0, 10.0)
    external_torque_range: tuple = (-10.0, 10.0)


@configclass
class MyDogHistRewardWeights:
    """HIM 版本的奖励权重配置 - 禁用速度跟踪，专注站立平衡"""

    # 通用
    is_terminated: float = -200

    # 速度跟踪奖励 - 全部禁用
    track_lin_vel_xy_exp: float = 0.0
    track_ang_vel_z_exp: float = 0.0
    upward: float = 3.0  # 【关键】直立奖励，防止机器人蠕动

    # 根部惩罚
    lin_vel_z_l2: float = -2.0
    ang_vel_xy_l2: float = -0.05
    flat_orientation_l2: float = 0.0
    base_height_l2: float = 0.0  # 高度惩罚
    body_lin_acc_l2: float = 0.0

    # 关节惩罚
    joint_torques_l2: float = -1e-5
    joint_torques_wheel_l2: float = 0.0
    joint_vel_l2: float = 0.0
    joint_vel_wheel_l2: float = 0.0
    joint_acc_l2: float = -2.5e-7
    joint_acc_wheel_l2: float = 0.0
    joint_pos_limits: float = -4.0
    joint_vel_limits: float = 0.0
    joint_power: float = -2e-5
    stand_still: float = 0.0  # 禁用，因为没有速度命令
    joint_pos_penalty: float = 0.0  # 禁用，因为没有速度命令
    wheel_vel_penalty: float = 0.0
    joint_mirror: float = 0.0

    # 动作惩罚
    action_rate_l2: float = -0.01

    # 接触惩罚
    undesired_contacts: float = -5.0
    contact_forces: float = -6e-4

    # 其他奖励 - 全部禁用
    feet_air_time: float = 0.0
    feet_contact: float = 0.0
    feet_contact_without_cmd: float = 1.0  # 禁用，因为没有速度命令
    feet_stumble: float = 0.0
    feet_slide: float = 0.0
    feet_height: float = 0.0
    feet_height_body: float = 0.0
    feet_gait: float = 0.0


@configclass
class MyDogHistObservationsCfg(ObservationsCfg):
    """
    HIM 版本的观测配置 - 带 5 帧历史观测
    
    观测结构（每帧 57 维 × 5 帧 = 285 维）：
    - base_ang_vel: 3D × 5 = 15D
    - projected_gravity: 3D × 5 = 15D  
    - velocity_commands: 3D × 5 = 15D
    - joint_pos: 16D × 5 = 80D (包含 4 个轮子位置置零)
    - joint_vel: 16D × 5 = 80D
    - actions: 16D × 5 = 80D
    
    注意：HIM 框架需要观测顺序为 [var1_history, var2_history, ...]
    而 Actor 期望的顺序为 [timestep_0_all_vars, timestep_1_all_vars, ...]
    这个转换在 HIMOnPolicyRunner 中的 reshape_isaac_to_him() 函数完成
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 观测配置 - 带 5 帧历史信息"""
        
        # 基座角速度 - 5帧历史 (3维 × 5帧 = 15维)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,  # 与 HIMLoco 论文一致
        )

        # 投影重力向量 - 5帧历史 (3维 × 5帧 = 15维)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 - 5帧历史 (3维 × 5帧 = 15维)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        # 关节位置观测 - 5帧历史 (16关节 × 5帧 = 80维)
        # 注意：轮子位置会被置零（因为轮子是无限旋转的）
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节速度观测 - 5帧历史 (16关节 × 5帧 = 80维)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,  # 速度缩放
        )

        # 上一步动作 - 5帧历史 (16动作 × 5帧 = 80维)
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5  # HIM 核心：5 帧历史观测

    @configclass
    class CriticCfg(ObsGroup):
        """
        Critic 观测配置 - 带 5 帧历史信息
        
        Critic 需要额外包含 base_lin_vel（速度真值），用于：
        1. 计算 value function（更准确的状态估计）
        2. 监督 Estimator 的速度预测（SwAV 训练）
        """

        # 基座线速度 - 5帧历史 (3维 × 5帧 = 15维) 
        # 【重要】这是 HIM 训练的监督信号！
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 基座角速度 - 5帧历史 (3维 × 5帧 = 15维)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 投影重力向量 - 5帧历史 (3维 × 5帧 = 15维)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 - 5帧历史 (3维 × 5帧 = 15维)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节位置观测 - 5帧历史 (16关节 × 5帧 = 80维)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节速度观测 - 5帧历史 (16关节 × 5帧 = 80维)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 上一步动作 - 5帧历史 (16动作 × 5帧 = 80维)
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False  # Critic 不加噪声
            self.concatenate_terms = True
            self.history_length = 5  # 与 Policy 一致

    @configclass
    class HeightScanCfg(ObsGroup):
        """
        高度扫描观测组 - 分离出来便于 HIM Runner 处理
        
        注意：height_scan 不带历史，因为地形信息相对稳定
        """
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.05},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 实例化观测组
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    height_scan_group: HeightScanCfg = HeightScanCfg()


@configclass
class MyDogHistActionsCfg(ActionsCfg):
    """HIM 版本的动作配置 - 与标准版相同"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale={
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_foot_joint", "FL_foot_joint",
            "RR_foot_joint", "RL_foot_joint",
        ],
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class MyDogHistRewardsCfg(RewardsCfg):
    """HIM 版本的奖励配置"""

    # 轮子相关的额外奖励项
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )


@configclass
class MyDogHistActuatorGains:
    """执行器 PD 增益配置"""
    hip_stiffness: float = 100.0
    hip_damping: float = 5.0
    thigh_stiffness: float = 100.0
    thigh_damping: float = 5.0
    calf_stiffness: float = 100.0
    calf_damping: float = 5.0
    wheel_stiffness: float = 0.0
    wheel_damping: float = 1.0


@configclass
class MyDogHistRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    HIM 版本的环境配置 - 带 5 帧历史观测
    
    与标准版本的主要区别：
    1. 观测带 5 帧历史 (history_length=5)
    2. height_scan 分离到独立观测组
    3. Critic 包含 base_lin_vel 用于监督 Estimator
    4. 奖励权重参考 HIMLoco 论文调整
    """

    # 使用 HIM 版本的配置
    observations: MyDogHistObservationsCfg = MyDogHistObservationsCfg()
    actions: MyDogHistActionsCfg = MyDogHistActionsCfg()
    rewards: MyDogHistRewardsCfg = MyDogHistRewardsCfg()
    reward_weights: MyDogHistRewardWeights = MyDogHistRewardWeights()
    event_params: MyDogHistEventParams = MyDogHistEventParams()
    command_params: MyDogHistCommandParams = MyDogHistCommandParams()
    actuator_gains: MyDogHistActuatorGains = MyDogHistActuatorGains()

    # Link 名称配置
    base_link_name = "base_link"
    foot_link_name = ".*_foot"

    # 关节名称配置
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        # 初始状态配置
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.55)
        self.scene.robot.init_state.joint_pos = {
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": -0.9,
            "FR_calf_joint": 1.8,
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 2.2,
            "RR_calf_joint": 1.8,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": -2.2,
            "RL_calf_joint": -1.8,
            ".*_foot_joint": 0.0,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}

        # ------------------------------Observations------------------------------
        # 配置关节观测
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        if "wheel_asset_cfg" in self.observations.policy.joint_pos.params:
            self.observations.policy.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        if "wheel_asset_cfg" in self.observations.critic.joint_pos.params:
            self.observations.critic.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        e = self.event_params
        
        # 复位基座随机化
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": e.reset_base_pose_range_x,
                "y": e.reset_base_pose_range_y,
                "z": e.reset_base_pose_range_z,
                "roll": e.reset_base_pose_range_roll,
                "pitch": e.reset_base_pose_range_pitch,
                "yaw": e.reset_base_pose_range_yaw,
            },
            "velocity_range": {
                "x": e.reset_base_velocity_range_x,
                "y": e.reset_base_velocity_range_y,
                "z": e.reset_base_velocity_range_z,
                "roll": e.reset_base_velocity_range_roll,
                "pitch": e.reset_base_velocity_range_pitch,
                "yaw": e.reset_base_velocity_range_yaw,
            },
        }
        
        # 质量随机化
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        
        # 质心随机化
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]

        # 外力/力矩随机化 - 禁用（避免空 body_names 解析错误）
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        # ------------------------------Rewards------------------------------
        w = self.reward_weights
        
        # 通用
        self.rewards.is_terminated.weight = w.is_terminated
        
        # 速度跟踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = w.track_lin_vel_xy_exp
        self.rewards.track_ang_vel_z_exp.weight = w.track_ang_vel_z_exp
        self.rewards.upward.weight = w.upward
        
        # 根部惩罚
        self.rewards.lin_vel_z_l2.weight = w.lin_vel_z_l2
        self.rewards.ang_vel_xy_l2.weight = w.ang_vel_xy_l2
        self.rewards.flat_orientation_l2.weight = w.flat_orientation_l2
        self.rewards.base_height_l2.weight = w.base_height_l2
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = w.body_lin_acc_l2
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        
        # 关节惩罚
        self.rewards.joint_torques_l2.weight = w.joint_torques_l2
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = w.joint_torques_wheel_l2
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = w.joint_vel_l2
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = w.joint_vel_wheel_l2
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = w.joint_acc_l2
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = w.joint_acc_wheel_l2
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = w.joint_pos_limits
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = w.joint_vel_limits
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = w.joint_power
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = w.stand_still
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = w.joint_pos_penalty
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = w.wheel_vel_penalty
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = w.joint_mirror
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        
        # 动作惩罚
        self.rewards.action_rate_l2.weight = w.action_rate_l2
        
        # 接触惩罚
        self.rewards.undesired_contacts.weight = w.undesired_contacts
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        
        self.rewards.contact_forces.weight = w.contact_forces
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 其他奖励
        self.rewards.feet_air_time.weight = w.feet_air_time
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = w.feet_contact
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["expect_contact_num"] = 4
        self.rewards.feet_contact_without_cmd.weight = w.feet_contact_without_cmd
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = w.feet_stumble
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = w.feet_slide
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = w.feet_height
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = w.feet_height_body
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = w.feet_gait
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )

        # 删除权重为 0 的奖励
        if self.__class__.__name__ == "MyDogHistRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculum------------------------------
        # 禁用所有课程学习
       
        
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels = None
        self.curriculum.disturbance_levels = None
        self.curriculum.mass_randomization_levels = None  
        self.curriculum.com_randomization_levels = None

        # ------------------------------Commands------------------------------
        # 禁用速度命令（设置为零范围）
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
