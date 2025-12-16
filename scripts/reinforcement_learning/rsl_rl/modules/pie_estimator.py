# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) Estimator
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 实现双层隐式-显式估计：
# 第一层：隐式(预测o_{t+1}) vs 显式(重建高度图m_t)
# 第二层：编码潜在向量(z_m, z_t) vs 显式物理量(v_hat, h_f_hat)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .pie_networks import (
    DepthEncoder,
    ProprioEncoder,
    CrossModalTransformer,
    MemoryGRU,
    MultiHeadDecoder,
)


@dataclass
class PIEEstimatorConfig:
    """PIE Estimator 配置"""

    # 输入维度
    num_depth_frames: int = 2       # H1: 深度图时序帧数
    depth_height: int = 58          # 深度图高度
    depth_width: int = 87           # 深度图宽度
    num_proprio_frames: int = 10    # H2: 本体感受历史帧数
    proprio_dim: int = 57           # 单帧本体感受维度 (MyDog: 57)

    # 编码器维度
    depth_output_dim: int = 32      # CNN输出维度
    proprio_output_dim: int = 64    # MLP输出维度

    # Transformer配置
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1

    # GRU配置
    gru_hidden_dim: int = 512
    gru_num_layers: int = 1

    # 输出维度
    latent_dim: int = 32            # z_m, z_t 维度
    velocity_dim: int = 3           # 基座速度维度
    foot_clearance_dim: int = 4     # 足部间隙维度 (4个足)
    height_map_dim: int = 187       # 高度扫描点数 (Isaac Lab height_scanner默认)

    # 损失权重
    lambda_kl: float = 0.01         # KL散度权重
    lambda_state: float = 1.0       # 状态预测损失权重 (隐式估计)
    lambda_map: float = 1.0         # 高度图重建损失权重 (显式估计)
    lambda_vel: float = 1.0         # 速度估计损失权重
    lambda_fc: float = 1.0          # 足部间隙估计损失权重

    # 优化器配置
    learning_rate: float = 1e-3


class PIEEstimator(nn.Module):
    """PIE Estimator - 双层隐式-显式估计器

    PIE框架的核心组件，融合深度图像和本体感受，输出：
    1. v_hat: 基座速度估计 (3D) - 显式物理量
    2. h_f_hat: 足部间隙估计 (4D) - 显式物理量
    3. z_m: 高度图编码 (latent_dim) - 编码潜在向量
    4. z_t: VAE隐变量 (latent_dim) - 用于隐式状态预测

    训练损失:
    L = λ_kl * KL(q(z_t) || N(0,1))
      + λ_state * MSE(o_{t+1}^hat, o_{t+1})  # 隐式估计
      + λ_map * MSE(m_hat, m)                 # 显式地形估计
      + λ_vel * MSE(v_hat, v)                 # 速度估计
      + λ_fc * MSE(h_f_hat, h_f)              # 足部间隙估计
    """

    def __init__(self, cfg: PIEEstimatorConfig):
        super().__init__()
        self.cfg = cfg

        # 深度图编码器
        self.depth_encoder = DepthEncoder(
            num_frames=cfg.num_depth_frames,
            output_dim=cfg.depth_output_dim,
        )

        # 本体感受编码器
        proprio_input_dim = cfg.num_proprio_frames * cfg.proprio_dim
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_input_dim,
            output_dim=cfg.proprio_output_dim,
        )

        # 跨模态Transformer
        self.transformer = CrossModalTransformer(
            depth_dim=cfg.depth_output_dim,
            proprio_dim=cfg.proprio_output_dim,
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_nhead,
            num_layers=cfg.transformer_layers,
            dropout=cfg.transformer_dropout,
        )

        # GRU记忆模块
        self.gru = MemoryGRU(
            input_dim=cfg.transformer_d_model,
            hidden_dim=cfg.gru_hidden_dim,
            num_layers=cfg.gru_num_layers,
        )

        # 多头解码器
        self.decoder = MultiHeadDecoder(
            input_dim=cfg.gru_hidden_dim,
            velocity_dim=cfg.velocity_dim,
            foot_clearance_dim=cfg.foot_clearance_dim,
            latent_dim=cfg.latent_dim,
            height_map_dim=cfg.height_map_dim,
            proprio_dim=cfg.proprio_dim,
        )

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.learning_rate,
        )

    def reset_hidden_state(self, batch_size: int, device: torch.device):
        """重置GRU隐状态"""
        self.gru.reset_hidden_state(batch_size, device)

    def reset_hidden_state_for_envs(self, env_ids: torch.Tensor):
        """为特定环境重置GRU隐状态"""
        self.gru.reset_hidden_state_for_envs(env_ids)

    def detach_hidden_state(self):
        """分离隐状态计算图"""
        self.gru.detach_hidden_state()

    def encode(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """编码深度图像和本体感受历史

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史

        Returns:
            outputs: 包含所有估计结果的字典（保留梯度）
        """
        # 深度图编码
        depth_features = self.depth_encoder(depth_images)  # [batch, 32]

        # 本体感受编码
        proprio_features = self.proprio_encoder(proprio_history)  # [batch, 64]

        # Transformer跨模态融合
        fused_features = self.transformer(depth_features, proprio_features)  # [batch, 64]

        # GRU记忆
        gru_output = self.gru(fused_features)  # [batch, 512]

        # 多头解码
        outputs = self.decoder(gru_output, training=self.training)

        return outputs

    def forward(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播（推理模式）

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史

        Returns:
            v_hat: [batch, 3] 速度估计
            h_f_hat: [batch, 4] 足部间隙估计
            z_m: [batch, latent_dim] 高度图编码
            z_t: [batch, latent_dim] VAE隐变量
        """
        with torch.no_grad():
            outputs = self.encode(depth_images, proprio_history)

        return (
            outputs["v_hat"].detach(),
            outputs["h_f_hat"].detach(),
            outputs["z_m"].detach(),
            outputs["z_t"].detach(),
        )

    def compute_loss(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        gt_velocity: torch.Tensor,
        gt_foot_clearance: torch.Tensor,
        gt_height_map: torch.Tensor,
        gt_next_proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算Estimator训练损失

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            gt_velocity: [batch, 3] 真实基座速度
            gt_foot_clearance: [batch, 4] 真实足部间隙
            gt_height_map: [batch, height_map_dim] 真实高度图
            gt_next_proprio: [batch, proprio_dim] 真实下一步本体感受

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 前向传播（训练模式，保留梯度）
        outputs = self.encode(depth_images, proprio_history)

        # 1. KL散度损失 (VAE正则化)
        kl_loss = self.decoder.vae_head.kl_divergence(
            outputs["mu"], outputs["logvar"]
        ).mean()

        # 2. 状态预测损失 (隐式估计) - 预测下一步本体感受
        state_loss = F.mse_loss(outputs["o_next_hat"], gt_next_proprio)

        # 3. 高度图重建损失 (显式估计)
        map_loss = F.mse_loss(outputs["m_hat"], gt_height_map)

        # 4. 速度估计损失
        vel_loss = F.mse_loss(outputs["v_hat"], gt_velocity)

        # 5. 足部间隙估计损失
        fc_loss = F.mse_loss(outputs["h_f_hat"], gt_foot_clearance)

        # 加权总损失
        total_loss = (
            self.cfg.lambda_kl * kl_loss
            + self.cfg.lambda_state * state_loss
            + self.cfg.lambda_map * map_loss
            + self.cfg.lambda_vel * vel_loss
            + self.cfg.lambda_fc * fc_loss
        )

        loss_dict = {
            "estimator/total_loss": total_loss.item(),
            "estimator/kl_loss": kl_loss.item(),
            "estimator/state_loss": state_loss.item(),
            "estimator/map_loss": map_loss.item(),
            "estimator/vel_loss": vel_loss.item(),
            "estimator/fc_loss": fc_loss.item(),
        }

        return total_loss, loss_dict

    def update(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        gt_velocity: torch.Tensor,
        gt_foot_clearance: torch.Tensor,
        gt_height_map: torch.Tensor,
        gt_next_proprio: torch.Tensor,
    ) -> Dict[str, float]:
        """更新Estimator参数

        Args:
            (同compute_loss)

        Returns:
            loss_dict: 各项损失的字典
        """
        self.optimizer.zero_grad()

        total_loss, loss_dict = self.compute_loss(
            depth_images=depth_images,
            proprio_history=proprio_history,
            gt_velocity=gt_velocity,
            gt_foot_clearance=gt_foot_clearance,
            gt_height_map=gt_height_map,
            gt_next_proprio=gt_next_proprio,
        )

        total_loss.backward()
        self.optimizer.step()

        return loss_dict

    def get_policy_inputs(
        self,
        depth_images: torch.Tensor,
        proprio_history: torch.Tensor,
        current_proprio: torch.Tensor,
    ) -> torch.Tensor:
        """获取Policy网络的输入

        将当前本体感受与Estimator输出拼接，作为Actor的输入。

        Args:
            depth_images: [batch, H1, H, W] 时序深度图像
            proprio_history: [batch, H2 * proprio_dim] 本体感受历史
            current_proprio: [batch, proprio_dim] 当前本体感受

        Returns:
            policy_input: [batch, proprio_dim + 3 + 4 + latent_dim * 2]
                         = [batch, 57 + 3 + 4 + 32 + 32] = [batch, 128]
        """
        v_hat, h_f_hat, z_m, z_t = self.forward(depth_images, proprio_history)

        # 拼接: o_t + v_hat + h_f_hat + z_m + z_t
        policy_input = torch.cat([
            current_proprio,  # [batch, 57]
            v_hat,            # [batch, 3]
            h_f_hat,          # [batch, 4]
            z_m,              # [batch, 32]
            z_t,              # [batch, 32]
        ], dim=-1)

        return policy_input

    @property
    def policy_input_dim(self) -> int:
        """Policy网络输入维度"""
        return (
            self.cfg.proprio_dim
            + self.cfg.velocity_dim
            + self.cfg.foot_clearance_dim
            + self.cfg.latent_dim * 2
        )


def create_pie_estimator(
    proprio_dim: int = 57,
    num_proprio_frames: int = 10,
    num_depth_frames: int = 2,
    latent_dim: int = 32,
    height_map_dim: int = 160,
    foot_clearance_dim: int = 4,
    learning_rate: float = 1e-3,
    **kwargs,
) -> PIEEstimator:
    """创建PIE Estimator的便捷函数

    Args:
        proprio_dim: 本体感受维度
        num_proprio_frames: 本体感受历史帧数 (H2)
        num_depth_frames: 深度图时序帧数 (H1)
        latent_dim: 隐变量维度
        height_map_dim: 高度图扫描点数
        foot_clearance_dim: 足部间隙维度
        learning_rate: 学习率

    Returns:
        PIEEstimator实例
    """
    cfg = PIEEstimatorConfig(
        proprio_dim=proprio_dim,
        num_proprio_frames=num_proprio_frames,
        num_depth_frames=num_depth_frames,
        latent_dim=latent_dim,
        height_map_dim=height_map_dim,
        foot_clearance_dim=foot_clearance_dim,
        learning_rate=learning_rate,
    )
    return PIEEstimator(cfg)
