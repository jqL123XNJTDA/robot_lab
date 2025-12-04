# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog 机器人的 RSL-RL PPO 配置

本文件包含两个版本：
1. MyDogRoughPPORunnerCfg / MyDogFlatPPORunnerCfg - 标准 PPO 版本
2. MyDogHistRoughPPORunnerCfg / MyDogHistFlatPPORunnerCfg - HIM 版本（带历史观测）
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# ==============================================================================
# 标准 PPO 版本 - 无历史观测
# ==============================================================================


@configclass
class MyDogRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """标准 PPO 配置 - Rough 地形"""
    
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
    """标准 PPO 配置 - Flat 地形"""
    
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "mydog_flat"


# ==============================================================================
# HIM 版本 - 带历史观测 (用于 train_him.py)
# ==============================================================================


@configclass
class MyDogHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    HIM 版本 PPO 配置 - Rough 地形
    
    关键配置项：
    1. num_steps_per_env = 200 (与 HIMLoco 论文一致)
    2. obs_groups 指定 policy 和 critic 使用的观测组
       - policy: ["policy"] - 只使用 policy 观测组
       - critic: ["critic", "height_scan_group"] - 使用 critic + 高度扫描
    3. max_grad_norm = 10.0 (与 HIMLoco 论文一致)
    
    注意：此配置需要配合 train_him.py 使用，该脚本会：
    1. 读取 history_length 自动配置 HIM 网络
    2. 使用 HIMOnPolicyRunner 替代标准 Runner
    3. 使用 HIMPPO 算法（包含 SwAV 训练）
    """
    
    num_steps_per_env = 200  # 与 HIMLoco 论文一致（4096 envs × 200 steps）
    max_iterations = 20000
    save_interval = 200
    experiment_name = "mydog_hist_rough"
    class_name = "OnPolicyRunner"  # 实际使用时会被 train_him.py 替换为 HIMOnPolicyRunner
    
    # 【关键】指定观测组分配
    # policy 组用于 Actor + Estimator 训练
    # critic 组用于 Value Function（包含特权信息）
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"]
    }

    # 续接训练参数
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # 网络配置 - 与 HIMLoco 论文一致
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # 算法配置 - 与 HIMLoco 论文一致
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # 与论文一致
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,  # 与论文一致
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,  # 【重要】与 HIMLoco 论文一致，不是 1.0
    )


@configclass
class MyDogHistFlatPPORunnerCfg(MyDogHistRoughPPORunnerCfg):
    """
    HIM 版本 PPO 配置 - Flat 地形
    
    用于：
    1. 初期在平坦地形上预训练（更容易收敛）
    2. 测试和调试 HIM 框架
    """
    
    max_iterations = 10000
    experiment_name = "mydog_hist_flat"
    class_name = "OnPolicyRunner"
    
    # Flat 环境不需要高度扫描
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"]  # 不包含 height_scan_group
    }
