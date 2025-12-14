# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Helios Leg RSL-RL PPO 训练配置。

提供两组配置：
- 标准 PPO: HeliosLegRoughPPORunnerCfg, HeliosLegFlatPPORunnerCfg
- HIM 风格 PPO: HeliosLegHistRoughPPORunnerCfg, HeliosLegHistFlatPPORunnerCfg
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# 标准 PPO 配置
##


@configclass
class HeliosLegRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """标准 PPO 配置 - Rough 环境"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "helios_leg_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeliosLegFlatPPORunnerCfg(HeliosLegRoughPPORunnerCfg):
    """标准 PPO 配置 - Flat 环境"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 8000
        self.experiment_name = "helios_leg_flat"


##
# HIM 风格 PPO 配置 - 带历史观测
##


@configclass
class HeliosLegHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM 风格 PPO 配置 - Rough 环境

    基于 HIM (Hybrid Internal Model) 论文参数：
    - num_steps_per_env=200 (HIM: 4096×200)
    - max_grad_norm=10.0
    - obs_groups 配置 policy/critic 观测组分离
    """

    num_steps_per_env = 200  # HIM 论文: 4096×200
    max_iterations = 20000
    save_interval = 100
    experiment_name = "helios_leg_hist_rough"
    # HIM 风格: obs_groups 分组配置
    # policy 只用 policy 观测组（不含 base_lin_vel）
    # critic 用 critic + height_scan_group（含 base_lin_vel + 高度扫描）
    obs_groups = {"policy": ["policy"], "critic": ["critic", "height_scan_group"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,  # HIM 论文: 10.0
    )


@configclass
class HeliosLegHistFlatPPORunnerCfg(HeliosLegHistRoughPPORunnerCfg):
    """HIM 风格 PPO 配置 - Flat 环境

    Flat 环境无高度扫描，critic 只用 critic 观测组
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 8000
        self.experiment_name = "helios_leg_hist_flat"
        # Flat 环境无 height_scan_group，critic 只用 critic 观测组
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class HeliosLegJumpPPORunnerCfg(HeliosLegFlatPPORunnerCfg):
    """Helios 跳跃训练 PPO 配置

    基于 Flat 配置，调整学习参数以适应跳跃任务。
    """

    def __post_init__(self):
        super().__post_init__()

        # 跳跃训练需要更多迭代
        self.max_iterations = 50000
        self.experiment_name = "helios_leg_jump"

        # # 使用较低的学习率（跳跃任务更复杂）
        # self.algorithm.learning_rate = 1.0e-4

        # # 增加梯度裁剪（防止跳跃时的大梯度）
        # self.algorithm.max_grad_norm = 1.0

        # # 稍微增加熵系数（鼓励探索跳跃动作）
        # self.algorithm.entropy_coef = 0.015


@configclass
class HeliosLegJumpLowAssistPPORunnerCfg(HeliosLegJumpPPORunnerCfg):
    """Helios 跳跃训练 PPO 配置（低辅助推力阶段）"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "helios_leg_jump_low_assist"


@configclass
class HeliosLegJumpNoAssistPPORunnerCfg(HeliosLegJumpPPORunnerCfg):
    """Helios 跳跃训练 PPO 配置（无辅助推力阶段）"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "helios_leg_jump_no_assist"
