# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) 环境配置
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# MyDog PIE环境配置

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.myrobots_mydogw.rough_env_cfg import (
    MyDogActionsCfg,
    MyDogRoughEnvCfg,
)


# ==============================================================================
# PIE Parkour 地形配置 - 使用Isaac Lab原生API
# ==============================================================================

@configclass
class PIEParkourTerrainsCfg(TerrainGeneratorCfg):
    """PIE Parkour地形配置

    针对MyDog (体高~0.45m，体长~0.4m) 适配的Parkour地形：
    - 台阶: 0.1-0.3m (约0.5-0.7x体高)
    - 障碍箱: 0.05-0.2m
    - 粗糙地形: noise 0.02-0.1m
    """

    size = (8.0, 8.0)
    border_width = 20.0
    num_rows = 10
    num_cols = 20
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75
    use_cache = False
    curriculum = True
    difficulty_range = (0.0, 1.0)


# ==============================================================================
# PIE 观测配置
# ==============================================================================

@configclass
class PIEObservationsCfg:
    """PIE观测配置

    PIE论文观测结构：
    - Policy观测: 本体感受历史 H2=10帧 (无深度图，深度图在Runner中单独处理)
    - Critic观测: 特权信息（真实速度、高度图等）

    本体感受 o_t (57D for MyDog):
    - 基座角速度: 3D
    - 投影重力: 3D
    - 速度命令: 3D
    - 关节位置: 16D (轮子位置=0)
    - 关节速度: 16D
    - 上一步动作: 16D

    Policy输入总维度: 57 × 10 = 570D (历史) + 当前帧57D = 627D
    Runner中会拼接Estimator输出 (3+4+32+32=71D) → 最终Actor输入
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy观测配置 - 本体感受 (无base_lin_vel)

        PIE论文中Policy只用本体感受，深度图由Estimator处理。
        使用H2=10帧历史。
        """

        # 基座角速度 [3D]
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        # 投影重力 [3D]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 [3D]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节位置 [16D] - 轮子位置设为0
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

        # 关节速度 [16D]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # 上一步动作 [16D]
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # PIE: 历史由Runner管理，环境只返回单帧
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):
        """Critic观测配置 - 特权信息

        PIE Critic使用真实的特权观测:
        - 真实基座速度 (3D)
        - 本体感受 (57D)
        - 高度图 (从height_scanner)
        """

        # 真实基座线速度 [3D] - 特权信息
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 基座角速度 [3D]
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 投影重力 [3D]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 [3D]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节位置 [16D]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节速度 [16D]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 上一步动作 [16D]
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 高度扫描 [187D] - 特权地形信息
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.05},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False  # Critic不加噪声
            self.concatenate_terms = True
            self.history_length = 1  # Critic不使用历史

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ==============================================================================
# PIE 奖励配置 - 论文 Table I
# ==============================================================================

@configclass
class PIERewardsCfg:
    """PIE奖励配置 - 基于论文Table I

    核心奖励项：
    - 速度跟踪: lin_vel_tracking, ang_vel_tracking
    - 姿态惩罚: lin_vel_z, ang_vel_xy, orientation
    - 关节惩罚: joint_acc, joint_power
    - 碰撞惩罚: collision
    - 动作平滑: action_rate, smoothness
    """

    # ===== 速度跟踪奖励 =====
    # r = exp{-4(v_cmd - v)²}
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # r = exp{-4(ω_cmd - ω)²}
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ===== 姿态惩罚 =====
    # r = -v_z²
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    # r = -ω_xy²
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # r = -|g|² (投影重力)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # ===== 关节惩罚 =====
    # r = -θ̈² (关节加速度)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # r = -|τ||θ̇| (关节功率)
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # 关节力矩惩罚
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # 关节位置限制
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # ===== 碰撞惩罚 =====
    # r = -n_collision
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold": 1.0,
        },
    )

    # 接触力过大惩罚
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 500.0,
        },
    )

    # ===== 动作惩罚 =====
    # r = -(a_t - a_{t-1})² (动作变化率)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # r = -(a_t - 2*a_{t-1} + a_{t-2})² (动作平滑度)
    # 注意: 这需要存储更多历史，简化为action_rate的平方
    action_rate_l2_2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # ===== 其他奖励 =====
    # 站立稳定
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # 保持向上姿态
    upward = RewTerm(
        func=mdp.upward,
        weight=0.5,
    )

    # 终止惩罚
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-0.0,
    )


# ==============================================================================
# PIE 环境配置 - 平地版本 (初始训练)
# ==============================================================================

@configclass
class MyDogPIEFlatEnvCfg(MyDogRoughEnvCfg):
    """MyDog PIE 平地环境配置

    用于PIE初始训练阶段：
    - 平地地形
    - 添加深度相机
    - PIE观测配置 (H2=10帧历史)
    - PIE奖励配置
    """

    # 使用PIE观测配置
    observations: PIEObservationsCfg = PIEObservationsCfg()

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # ------------------------------Episode 长度------------------------------
        # PIE训练使用更短的episode，便于更快获得反馈
        self.episode_length_s = 10.0  # 10秒 (父类是20秒)

        # ------------------------------Scene 场景------------------------------
        # 使用平地地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 添加深度相机
        terrain_prim_path = "/World/ground"
        self.scene.depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            update_period=0.1,  # 10Hz (PIE论文)
            data_types=["distance_to_image_plane"],
            debug_vis=False,
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.6, 0.0, 0.1),  # 前视相机位置
                rot=(0.9659, 0.0, 0.2588, 0.0),  # 俯视30度
                convention="world",
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=47.5,  # FOV ≈ 87°
                width=106,
                height=60,
            ),
            max_distance=3.0,  # 与clip_range(0.3, 3.0)一致
            mesh_prim_paths=[terrain_prim_path],
        )

        # ------------------------------Observations 观测------------------------------
        # 设置关节名称
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Rewards 奖励------------------------------
        # PIE奖励配置 - 覆盖父类
        # 速度跟踪
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # 姿态惩罚
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0

        # 关节惩罚
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names

        # 碰撞惩罚
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]

        # 动作平滑
        self.rewards.action_rate_l2.weight = -0.01

        # 其他
        self.rewards.upward.weight = 0.5
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names

        # 自动移除权重为0的奖励
        if self.__class__.__name__ == "MyDogPIEFlatEnvCfg":
            self.disable_zero_weight_rewards()


# ==============================================================================
# PIE 环境配置 - 粗糙地形版本 (进阶训练)
# ==============================================================================

@configclass
class MyDogPIERoughEnvCfg(MyDogPIEFlatEnvCfg):
    """MyDog PIE 粗糙地形环境配置

    用于PIE进阶训练：
    - 粗糙地形 (Isaac Lab原生)
    - 课程学习
    """

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # 恢复粗糙地形
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5

        # 更新深度相机的mesh_prim_paths
        self.scene.depth_camera.mesh_prim_paths = ["/World/ground"]

        # 启用高度扫描
        # Critic已经包含height_scan

        # 课程学习
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # 自动移除权重为0的奖励
        if self.__class__.__name__ == "MyDogPIERoughEnvCfg":
            self.disable_zero_weight_rewards()


# ==============================================================================
# PIE 环境配置 - Parkour地形版本 (高级训练)
# ==============================================================================

@configclass
class MyDogPIEParkourEnvCfg(MyDogPIERoughEnvCfg):
    """MyDog PIE Parkour环境配置

    用于PIE Parkour训练：
    - 自定义Parkour地形（台阶、障碍、间隙等）
    - 更激进的课程学习
    """

    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # 使用Parkour地形配置
        # 注意: Isaac Lab原生不包含Gap地形，需要自定义
        from isaaclab.terrains import terrain_generator as tg

        parkour_terrain_cfg = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                # 台阶地形 - MyDog适配
                "pyramid_stairs": tg.MeshPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.1, 0.25),  # MyDog体高0.45m的0.22-0.55倍
                    step_width=0.3,
                    platform_width=1.0,
                    border_width=0.0,
                ),
                # 倒金字塔台阶
                "inverted_pyramid_stairs": tg.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.1, 0.2),
                    step_width=0.3,
                    platform_width=1.0,
                    border_width=0.0,
                ),
                # 随机箱子障碍
                "boxes": tg.MeshRandomGridTerrainCfg(
                    proportion=0.2,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.2),  # 0.11-0.44倍体高
                    platform_width=2.0,
                ),
                # 粗糙地形
                "random_rough": tg.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.02, 0.1),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                # 平地（基础训练/恢复）
                "flat": tg.MeshPlaneTerrainCfg(
                    proportion=0.2,
                ),
            },
        )

        self.scene.terrain.terrain_generator = parkour_terrain_cfg

        # 更激进的奖励配置
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.undesired_contacts.weight = -15.0

        # 自动移除权重为0的奖励
        if self.__class__.__name__ == "MyDogPIEParkourEnvCfg":
            self.disable_zero_weight_rewards()
