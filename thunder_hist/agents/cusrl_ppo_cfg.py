# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class ThunderHistRoughTrainerCfg(TrainerCfg):
    """Thunder历史观测配置的CusRL训练器 - 针对历史观测优化网络结构"""

    max_iterations = 20000
    save_interval = 50
    experiment_name = "thunder_hist_rough"

    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[768, 512, 256],  # Policy观测: 261维
                activation_fn="ELU",
                ends_with_activation=True,
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[768, 512, 256],  # Critic观测: 487维
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
class ThunderHistFlatTrainerCfg(ThunderHistRoughTrainerCfg):
    """Thunder历史观测平坦地形CusRL训练配置"""

    max_iterations = 5000
    experiment_name = "thunder_hist_flat"
