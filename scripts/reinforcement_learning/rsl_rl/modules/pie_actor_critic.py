# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) Actor-Critic
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# Actor输入: o_t + v_hat + h_f_hat + z_m + z_t
# Critic输入: s_t = [o_t, v_t(真实), m_t(高度图)] (特权观测)

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional

from .pie_estimator import PIEEstimator, PIEEstimatorConfig


class PIEActorCritic(nn.Module):
    """PIE Actor-Critic 网络

    非对称Actor-Critic架构：
    - Actor: 使用Estimator输出的估计值作为输入
    - Critic: 使用真实的特权观测（真实速度、高度图等）

    Actor输入维度: proprio_dim + 3 + 4 + latent_dim * 2
                 = 57 + 3 + 4 + 32 + 32 = 128 (MyDog)
    Critic输入维度: critic_obs_dim (包含特权信息)
    """

    def __init__(
        self,
        # Estimator配置
        estimator_cfg: PIEEstimatorConfig,
        # Actor配置
        num_actions: int,
        actor_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        # Critic配置
        critic_obs_dim: int = 247,  # 特权观测维度: 60(proprio+base_lin_vel) + 187(height_scan)
        critic_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        # 通用配置
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.init_noise_std = init_noise_std

        # PIE Estimator
        self.estimator = PIEEstimator(estimator_cfg)

        # 选择激活函数
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ELU()

        # Actor网络
        actor_input_dim = self.estimator.policy_input_dim
        actor_layers = []
        prev_dim = actor_input_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))

        self.actor = nn.Sequential(*actor_layers)

        # Critic网络
        critic_layers = []
        prev_dim = critic_obs_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))

        self.critic = nn.Sequential(*critic_layers)

        # 可学习的动作标准差
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # 动作分布（在forward时更新）
        self.distribution: Optional[Normal] = None

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化Actor和Critic权重"""
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """重置Estimator的GRU隐状态

        Args:
            env_ids: 需要重置的环境索引，None表示全部重置
        """
        if env_ids is None:
            # 重置所有环境
            pass  # 在下次forward时自动初始化
        else:
            self.estimator.reset_hidden_state_for_envs(env_ids)

    def update_distribution(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
    ):
        """更新动作分布

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受
        """
        # 获取Estimator输出并拼接为Policy输入
        policy_input = self.estimator.get_policy_inputs(
            depth_images, proprio_history, current_proprio
        )

        # Actor前向传播
        mean = self.actor(policy_input)

        # 更新分布
        self.distribution = Normal(mean, self.std)

    def act(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
    ) -> torch.Tensor:
        """采样动作（训练模式）

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受

        Returns:
            actions: [batch, num_actions] 采样的动作
        """
        self.update_distribution(depth_images, proprio_history, current_proprio)
        return self.distribution.sample()

    def act_inference(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
    ) -> torch.Tensor:
        """推理模式获取动作（确定性）

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受

        Returns:
            actions: [batch, num_actions] 确定性动作（均值）
        """
        self.update_distribution(depth_images, proprio_history, current_proprio)
        return self.distribution.mean

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """计算状态价值

        Args:
            critic_obs: [batch, critic_obs_dim] Critic特权观测

        Returns:
            value: [batch, 1] 状态价值
        """
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率

        Args:
            actions: [batch, num_actions] 动作

        Returns:
            log_prob: [batch] 对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        """动作均值"""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """动作标准差"""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """动作熵"""
        return self.distribution.entropy().sum(dim=-1)


class PIEActorCriticRecurrent(PIEActorCritic):
    """带循环Critic的PIE Actor-Critic

    可选的扩展：Critic也使用历史观测。
    本实现中Critic仍然使用单步观测，但保留此类以便未来扩展。
    """

    def __init__(
        self,
        estimator_cfg: PIEEstimatorConfig,
        num_actions: int,
        actor_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        critic_obs_dim: int = 220,
        critic_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__(
            estimator_cfg=estimator_cfg,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_obs_dim=critic_obs_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )


def create_pie_actor_critic(
    # Estimator配置
    proprio_dim: int = 57,
    num_proprio_frames: int = 10,
    num_depth_frames: int = 2,
    latent_dim: int = 32,
    height_map_dim: int = 187,  # 高度扫描点数 (Isaac Lab height_scanner默认)
    foot_clearance_dim: int = 4,
    estimator_lr: float = 1e-3,
    # Actor配置
    num_actions: int = 16,
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128),
    # Critic配置
    critic_obs_dim: int = 247,  # 特权观测维度: 60(proprio+base_lin_vel) + 187(height_scan)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128),
    # 通用配置
    activation: str = "elu",
    init_noise_std: float = 1.0,
) -> PIEActorCritic:
    """创建PIE Actor-Critic的便捷函数

    Args:
        proprio_dim: 本体感受维度
        num_proprio_frames: 本体感受历史帧数 (H2)
        num_depth_frames: 深度图时序帧数 (H1)
        latent_dim: 隐变量维度
        height_map_dim: 高度图扫描点数
        foot_clearance_dim: 足部间隙维度
        estimator_lr: Estimator学习率
        num_actions: 动作维度
        actor_hidden_dims: Actor隐藏层维度
        critic_obs_dim: Critic观测维度
        critic_hidden_dims: Critic隐藏层维度
        activation: 激活函数
        init_noise_std: 初始动作标准差

    Returns:
        PIEActorCritic实例
    """
    estimator_cfg = PIEEstimatorConfig(
        proprio_dim=proprio_dim,
        num_proprio_frames=num_proprio_frames,
        num_depth_frames=num_depth_frames,
        latent_dim=latent_dim,
        height_map_dim=height_map_dim,
        foot_clearance_dim=foot_clearance_dim,
        learning_rate=estimator_lr,
    )

    return PIEActorCritic(
        estimator_cfg=estimator_cfg,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_obs_dim=critic_obs_dim,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
    )
