# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) Rollout Storage
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 扩展存储以支持：
# - 时序深度图像
# - 本体感受历史
# - 足部间隙真值
# - 下一步本体感受（用于隐式状态预测监督）

import torch
from typing import Generator, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PIETransition:
    """PIE单步转移数据"""

    # Actor观测
    depth_images: torch.Tensor          # [batch, H1, H, W] 时序深度图
    proprio_history: torch.Tensor       # [batch, H2 * proprio_dim] 本体感受历史
    current_proprio: torch.Tensor       # [batch, proprio_dim] 当前本体感受

    # Critic观测
    critic_observations: torch.Tensor   # [batch, critic_dim] 特权观测

    # Estimator监督信号
    gt_velocity: torch.Tensor           # [batch, 3] 真实基座速度
    gt_foot_clearance: torch.Tensor     # [batch, 4] 真实足部间隙
    gt_height_map: torch.Tensor         # [batch, height_map_dim] 真实高度图
    next_proprio: torch.Tensor          # [batch, proprio_dim] 下一步本体感受

    # 标准RL数据
    actions: torch.Tensor               # [batch, action_dim] 动作
    rewards: torch.Tensor               # [batch] 奖励
    dones: torch.Tensor                 # [batch] 终止标志
    values: torch.Tensor                # [batch] 价值估计
    actions_log_prob: torch.Tensor      # [batch] 动作对数概率

    # 用于PPO的额外数据
    action_mean: torch.Tensor           # [batch, action_dim] 动作均值
    action_sigma: torch.Tensor          # [batch, action_dim] 动作标准差


class PIERolloutStorage:
    """PIE Rollout Storage

    存储PIE训练所需的所有数据，包括：
    - 时序深度图像
    - 本体感受历史
    - Estimator监督信号
    - 标准PPO数据
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        # 深度图配置
        num_depth_frames: int = 2,
        depth_height: int = 58,
        depth_width: int = 87,
        # 本体感受配置
        num_proprio_frames: int = 10,
        proprio_dim: int = 57,
        # Critic配置
        critic_obs_dim: int = 220,
        # Estimator监督配置
        height_map_dim: int = 160,
        foot_clearance_dim: int = 4,
        # 动作配置
        action_dim: int = 16,
        # 设备
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device

        # 维度
        self.num_depth_frames = num_depth_frames
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.num_proprio_frames = num_proprio_frames
        self.proprio_dim = proprio_dim
        self.critic_obs_dim = critic_obs_dim
        self.height_map_dim = height_map_dim
        self.foot_clearance_dim = foot_clearance_dim
        self.action_dim = action_dim

        # 当前步骤
        self.step = 0

        # 预分配存储空间
        self._allocate_storage()

    def _allocate_storage(self):
        """预分配存储张量"""
        N = self.num_transitions_per_env
        E = self.num_envs

        # Actor观测
        self.depth_images = torch.zeros(
            N, E, self.num_depth_frames, self.depth_height, self.depth_width,
            device=self.device
        )
        self.proprio_history = torch.zeros(
            N, E, self.num_proprio_frames * self.proprio_dim,
            device=self.device
        )
        self.current_proprio = torch.zeros(
            N, E, self.proprio_dim,
            device=self.device
        )

        # Critic观测
        self.critic_observations = torch.zeros(
            N, E, self.critic_obs_dim,
            device=self.device
        )

        # Estimator监督信号
        self.gt_velocity = torch.zeros(N, E, 3, device=self.device)
        self.gt_foot_clearance = torch.zeros(
            N, E, self.foot_clearance_dim, device=self.device
        )
        self.gt_height_map = torch.zeros(
            N, E, self.height_map_dim, device=self.device
        )
        self.next_proprio = torch.zeros(
            N, E, self.proprio_dim, device=self.device
        )

        # 标准RL数据
        self.actions = torch.zeros(N, E, self.action_dim, device=self.device)
        self.rewards = torch.zeros(N, E, 1, device=self.device)
        self.dones = torch.zeros(N, E, 1, device=self.device)
        self.values = torch.zeros(N, E, 1, device=self.device)
        self.actions_log_prob = torch.zeros(N, E, 1, device=self.device)

        # PPO额外数据
        self.action_mean = torch.zeros(N, E, self.action_dim, device=self.device)
        self.action_sigma = torch.zeros(N, E, self.action_dim, device=self.device)

        # GAE计算结果
        self.returns = torch.zeros(N, E, 1, device=self.device)
        self.advantages = torch.zeros(N, E, 1, device=self.device)

    def add_transitions(self, transition: PIETransition):
        """添加一步转移数据

        Args:
            transition: PIETransition数据
        """
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Storage is full. Call clear() first.")

        self.depth_images[self.step] = transition.depth_images
        self.proprio_history[self.step] = transition.proprio_history
        self.current_proprio[self.step] = transition.current_proprio
        self.critic_observations[self.step] = transition.critic_observations
        self.gt_velocity[self.step] = transition.gt_velocity
        self.gt_foot_clearance[self.step] = transition.gt_foot_clearance
        self.gt_height_map[self.step] = transition.gt_height_map
        self.next_proprio[self.step] = transition.next_proprio
        self.actions[self.step] = transition.actions
        self.rewards[self.step] = transition.rewards.unsqueeze(-1)
        self.dones[self.step] = transition.dones.unsqueeze(-1)
        self.values[self.step] = transition.values.unsqueeze(-1)
        self.actions_log_prob[self.step] = transition.actions_log_prob.unsqueeze(-1)
        self.action_mean[self.step] = transition.action_mean
        self.action_sigma[self.step] = transition.action_sigma

        self.step += 1

    def clear(self):
        """清空存储"""
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        """计算GAE和回报

        Args:
            last_values: [num_envs, 1] 最后一步的价值估计
            gamma: 折扣因子
            lam: GAE lambda
        """
        advantage = torch.zeros(self.num_envs, 1, device=self.device)

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step]
            delta = (
                self.rewards[step]
                + gamma * next_values * next_is_not_terminal
                - self.values[step]
            )
            advantage = delta + gamma * lam * next_is_not_terminal * advantage
            self.returns[step] = advantage + self.values[step]

        # 计算优势
        self.advantages = self.returns - self.values

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[dict, None, None]:
        """生成mini-batch

        Args:
            num_mini_batches: mini-batch数量
            num_epochs: 训练轮数

        Yields:
            batch_dict: 包含所有数据的字典
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # 展平所有数据
        # depth_images: [N*E, H1, H, W]
        depth_images = self.depth_images.reshape(-1, self.num_depth_frames,
                                                  self.depth_height, self.depth_width)
        proprio_history = self.proprio_history.reshape(
            -1, self.num_proprio_frames * self.proprio_dim
        )
        current_proprio = self.current_proprio.reshape(-1, self.proprio_dim)
        critic_observations = self.critic_observations.reshape(-1, self.critic_obs_dim)
        gt_velocity = self.gt_velocity.reshape(-1, 3)
        gt_foot_clearance = self.gt_foot_clearance.reshape(-1, self.foot_clearance_dim)
        gt_height_map = self.gt_height_map.reshape(-1, self.height_map_dim)
        next_proprio = self.next_proprio.reshape(-1, self.proprio_dim)
        actions = self.actions.reshape(-1, self.action_dim)
        values = self.values.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        old_actions_log_prob = self.actions_log_prob.reshape(-1, 1)
        advantages = self.advantages.reshape(-1, 1)
        old_action_mean = self.action_mean.reshape(-1, self.action_dim)
        old_action_sigma = self.action_sigma.reshape(-1, self.action_dim)

        for _ in range(num_epochs):
            # 打乱索引
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch_indices = indices[start:end]

                yield {
                    # Actor观测
                    "depth_images": depth_images[batch_indices],
                    "proprio_history": proprio_history[batch_indices],
                    "current_proprio": current_proprio[batch_indices],
                    # Critic观测
                    "critic_observations": critic_observations[batch_indices],
                    # Estimator监督
                    "gt_velocity": gt_velocity[batch_indices],
                    "gt_foot_clearance": gt_foot_clearance[batch_indices],
                    "gt_height_map": gt_height_map[batch_indices],
                    "next_proprio": next_proprio[batch_indices],
                    # PPO数据
                    "actions": actions[batch_indices],
                    "values": values[batch_indices],
                    "returns": returns[batch_indices],
                    "old_actions_log_prob": old_actions_log_prob[batch_indices],
                    "advantages": advantages[batch_indices],
                    "old_action_mean": old_action_mean[batch_indices],
                    "old_action_sigma": old_action_sigma[batch_indices],
                }
