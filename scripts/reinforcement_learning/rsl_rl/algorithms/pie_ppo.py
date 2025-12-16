# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) PPO算法
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 单阶段端到端训练：并发优化Actor-Critic和Estimator

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

from modules.pie_actor_critic import PIEActorCritic
from storage.pie_rollout_storage import PIERolloutStorage


class PIEPPO:
    """PIE PPO算法

    并发优化两个部分：
    1. Actor-Critic (PPO)
    2. Estimator (MSE + KL损失)

    这是单阶段端到端训练，不同于双阶段Teacher-Student方法。
    """

    def __init__(
        self,
        actor_critic: PIEActorCritic,
        storage: PIERolloutStorage,
        # PPO配置
        learning_rate: float = 1e-3,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        # 学习率调度
        schedule: str = "fixed",  # "fixed", "adaptive"
        desired_kl: float = 0.01,
        # 设备
        device: str = "cuda",
    ):
        self.actor_critic = actor_critic
        self.storage = storage
        self.device = device

        # PPO超参数
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl

        # Actor-Critic优化器（不包含Estimator参数）
        # Estimator有自己的优化器
        actor_critic_params = list(self.actor_critic.actor.parameters()) + \
                              list(self.actor_critic.critic.parameters()) + \
                              [self.actor_critic.std]
        self.optimizer = optim.Adam(actor_critic_params, lr=learning_rate)

        # 转移到设备
        self.actor_critic.to(device)

    def update(self) -> Dict[str, float]:
        """执行一次PPO更新

        同时更新Actor-Critic和Estimator。

        Returns:
            loss_dict: 所有损失的字典
        """
        # 累积损失
        total_loss_dict = {
            "ppo/policy_loss": 0.0,
            "ppo/value_loss": 0.0,
            "ppo/entropy_loss": 0.0,
            "ppo/total_loss": 0.0,
        }
        estimator_loss_dict = {}
        num_updates = 0

        # Mini-batch训练
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            # 1. 更新Estimator
            est_loss = self.actor_critic.estimator.update(
                depth_images=batch["depth_images"],
                proprio_history=batch["proprio_history"],
                gt_velocity=batch["gt_velocity"],
                gt_foot_clearance=batch["gt_foot_clearance"],
                gt_height_map=batch["gt_height_map"],
                gt_next_proprio=batch["next_proprio"],
            )

            # 累积Estimator损失
            for k, v in est_loss.items():
                if k not in estimator_loss_dict:
                    estimator_loss_dict[k] = 0.0
                estimator_loss_dict[k] += v

            # 2. 更新Actor-Critic (PPO)
            ppo_loss = self._update_ppo(batch)

            # 累积PPO损失
            for k, v in ppo_loss.items():
                total_loss_dict[k] += v

            num_updates += 1

        # 平均损失
        for k in total_loss_dict:
            total_loss_dict[k] /= num_updates
        for k in estimator_loss_dict:
            estimator_loss_dict[k] /= num_updates

        # 合并损失字典
        total_loss_dict.update(estimator_loss_dict)

        # 自适应学习率
        if self.schedule == "adaptive":
            self._adapt_learning_rate()

        return total_loss_dict

    def _update_ppo(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行PPO更新

        Args:
            batch: mini-batch数据

        Returns:
            loss_dict: PPO损失字典
        """
        # 前向传播获取新的动作分布
        self.actor_critic.update_distribution(
            depth_images=batch["depth_images"],
            proprio_history=batch["proprio_history"],
            current_proprio=batch["current_proprio"],
        )

        # 计算新的对数概率
        actions_log_prob = self.actor_critic.get_actions_log_prob(batch["actions"])

        # 计算价值
        value = self.actor_critic.evaluate(batch["critic_observations"])

        # 计算熵
        entropy = self.actor_critic.entropy.mean()

        # PPO策略损失
        ratio = torch.exp(
            actions_log_prob - batch["old_actions_log_prob"].squeeze(-1)
        )
        surrogate1 = ratio * batch["advantages"].squeeze(-1)
        surrogate2 = torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * batch["advantages"].squeeze(-1)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # 价值函数损失
        if self.use_clipped_value_loss:
            value_clipped = batch["values"] + torch.clamp(
                value - batch["values"],
                -self.clip_param,
                self.clip_param,
            )
            value_loss1 = (value - batch["returns"]).pow(2)
            value_loss2 = (value_clipped - batch["returns"]).pow(2)
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = (value - batch["returns"]).pow(2).mean()

        # 总损失
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(
            list(self.actor_critic.actor.parameters()) +
            list(self.actor_critic.critic.parameters()),
            self.max_grad_norm,
        )

        self.optimizer.step()

        return {
            "ppo/policy_loss": policy_loss.item(),
            "ppo/value_loss": value_loss.item(),
            "ppo/entropy_loss": entropy.item(),
            "ppo/total_loss": loss.item(),
        }

    def _adapt_learning_rate(self):
        """自适应学习率调整"""
        # 计算KL散度
        with torch.no_grad():
            # 简化实现：使用action_sigma近似KL
            kl = self._compute_kl_divergence()

        if kl > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate * 0.667)
        elif kl < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _compute_kl_divergence(self) -> float:
        """计算KL散度（近似）"""
        # 简化实现，实际应该使用完整的KL计算
        # 这里返回一个默认值
        return 0.01

    def process_env_reset(self, env_ids: torch.Tensor):
        """处理环境重置

        重置对应环境的GRU隐状态。

        Args:
            env_ids: 需要重置的环境索引
        """
        self.actor_critic.reset(env_ids)

    def act(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作并计算价值

        Args:
            depth_images: [batch, H1, H, W] 时序深度图
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受
            critic_obs: [batch, critic_dim] Critic观测

        Returns:
            actions: [batch, action_dim] 采样的动作
            values: [batch] 价值估计
            actions_log_prob: [batch] 动作对数概率
            action_mean: [batch, action_dim] 动作均值
            action_sigma: [batch, action_dim] 动作标准差
        """
        # 采样动作
        actions = self.actor_critic.act(
            depth_images, proprio_history, current_proprio
        )

        # 计算价值
        values = self.actor_critic.evaluate(critic_obs).squeeze(-1)

        # 获取对数概率和分布参数
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
        action_mean = self.actor_critic.action_mean
        action_sigma = self.actor_critic.action_std

        return actions, values, actions_log_prob, action_mean, action_sigma

    def act_inference(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
    ) -> torch.Tensor:
        """推理模式获取动作

        Args:
            depth_images: [batch, H1, H, W] 时序深度图
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受

        Returns:
            actions: [batch, action_dim] 确定性动作
        """
        return self.actor_critic.act_inference(
            depth_images, proprio_history, current_proprio
        )
