# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog RSL-RL PPO 训练配置。

提供两组配置：
- 标准 PPO: MyDogRoughPPORunnerCfg, MyDogFlatPPORunnerCfg
- HIM 风格 PPO: MyDogHistRoughPPORunnerCfg, MyDogHistFlatPPORunnerCfg
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# 标准 PPO 配置
##


@configclass
class MyDogRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """标准 PPO 配置 - Rough 环境"""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "mydog_rough"
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
class MyDogFlatPPORunnerCfg(MyDogRoughPPORunnerCfg):
    """标准 PPO 配置 - Flat 环境"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "mydog_flat"


##
# HIM 风格 PPO 配置 - 带历史观测
##


@configclass
class MyDogHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM 风格 PPO 配置 - Rough 环境

    基于 HIM (Hybrid Internal Model) 论文参数：
    - num_steps_per_env=200 (HIM: 4096×200)
    - max_grad_norm=10.0

    MyDog 观测维度：
    - policy_groups: ["policy"] (57×5=285 dims)
    - critic_groups: ["critic", "height_scan_group"] (60×5+187 dims)
    """

    num_steps_per_env = 200  # HIM 论文: 4096×200
    max_iterations = 20000
    save_interval = 100
    experiment_name = "mydog_hist_rough"
    # HIM 风格: obs_groups 分组配置（文档用途，实际由 HIMOnPolicyRunner 自动检测）
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
class MyDogHistFlatPPORunnerCfg(MyDogHistRoughPPORunnerCfg):
    """HIM 风格 PPO 配置 - Flat 环境

    Flat 环境无高度扫描，critic 只用 critic 观测组

    MyDog 观测维度：
    - policy_groups: ["policy"] (57×5=285 dims)
    - critic_groups: ["critic"] (60×5=300 dims)
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "mydog_hist_flat"
        # Flat 环境无 height_scan_group
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}