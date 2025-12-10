# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog 机器人的 CusRL PPO 配置

本文件包含两个版本：
1. MyDogRoughTrainerCfg / MyDogFlatTrainerCfg - 标准 PPO 版本
2. MyDogHistRoughTrainerCfg / MyDogHistFlatTrainerCfg - HIM 版本（带历史观测）
"""

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


# ==============================================================================
# 标准 PPO 版本 - 无历史观测
# ==============================================================================


@dataclass
class MyDogRoughTrainerCfg(TrainerCfg):
    """MyDog 标准 PPO 配置 - Rough 地形"""

    max_iterations = 20000
    save_interval = 100
    experiment_name = "mydog_rough"

    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128],
                activation_fn="ELU",
                ends_with_activation=True,
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128],
                activation_fn="ELU",
                ends_with_activation=True,
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.01),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class MyDogFlatTrainerCfg(MyDogRoughTrainerCfg):
    """MyDog 标准 PPO 配置 - Flat 地形"""

    max_iterations = 5000
    experiment_name = "mydog_flat"


# ==============================================================================
# HIM 版本 - 带历史观测（与 ThunderHist 一致）
# ==============================================================================


@dataclass
class MyDogHistRoughTrainerCfg(TrainerCfg):
    """MyDog HIM 版本 CusRL 配置 - Rough 地形（与 ThunderHistRoughTrainerCfg 一致）"""

    max_iterations = 20000
    save_interval = 50
    experiment_name = "mydog_hist_rough"

    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[768, 512, 256],  # Policy观测: 285维 (57*5)
                activation_fn="ELU",
                ends_with_activation=True,
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[768, 512, 256],  # Critic观测: 含特权信息
                activation_fn="ELU",
                ends_with_activation=True,
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),  # 稍微降低熵系数
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class MyDogHistFlatTrainerCfg(MyDogHistRoughTrainerCfg):
    """MyDog HIM 版本 CusRL 配置 - Flat 地形"""

    max_iterations = 5000
    experiment_name = "mydog_hist_flat"
