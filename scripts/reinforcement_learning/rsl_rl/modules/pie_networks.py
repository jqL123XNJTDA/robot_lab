# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) 子网络模块
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthEncoder(nn.Module):
    """CNN深度图编码器

    处理时序深度图像，提取视觉特征。
    输入: [batch, num_frames, H, W] = [batch, 2, 58, 87]
    输出: [batch, output_dim] = [batch, 32]

    参考Isaaclab_Parkour的DepthOnlyFCBackbone58x87架构
    """

    def __init__(
        self,
        num_frames: int = 2,  # H1=2 时序深度帧
        output_dim: int = 32,
        activation: str = "elu",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.output_dim = output_dim

        # 选择激活函数
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.ELU()

        # CNN编码器
        # 输入: [batch, num_frames, 58, 87]
        # Conv1: 58x87 -> 54x83 -> MaxPool -> 27x41
        # Conv2: 27x41 -> 25x39
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            act_fn,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            act_fn,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),  # 64 * 25 * 39 = 62400
            act_fn,
            nn.Linear(128, output_dim),
            act_fn,
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            depth_images: [batch, num_frames, H, W] 或 [batch, H, W]
                         时序深度图像，已归一化到[-0.5, 0.5]

        Returns:
            depth_features: [batch, output_dim] 深度特征
        """
        # 确保输入维度正确
        if depth_images.dim() == 3:
            # [batch, H, W] -> [batch, 1, H, W]
            depth_images = depth_images.unsqueeze(1)

        return self.encoder(depth_images)


class ProprioEncoder(nn.Module):
    """MLP本体感受编码器

    处理时序本体感受历史，提取状态特征。
    输入: [batch, H2 * obs_dim] = [batch, 10 * 57] = [batch, 570]
    输出: [batch, output_dim] = [batch, 64]
    """

    def __init__(
        self,
        input_dim: int = 570,  # H2=10, obs_dim=57 for MyDog
        output_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 选择激活函数
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.ELU()

        # MLP编码器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(act_fn)

        self.encoder = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, proprio_history: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            proprio_history: [batch, H2 * obs_dim] 拼接的本体感受历史

        Returns:
            proprio_features: [batch, output_dim] 本体感受特征
        """
        return self.encoder(proprio_history)


class CrossModalTransformer(nn.Module):
    """跨模态Transformer编码器

    融合深度特征和本体感受特征，实现跨模态推理。
    输入: depth_features [batch, depth_dim], proprio_features [batch, proprio_dim]
    输出: fused_features [batch, d_model]

    PIE论文中使用Transformer实现视觉和本体感受的交叉注意力
    """

    def __init__(
        self,
        depth_dim: int = 32,
        proprio_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 投影层：将两种模态映射到相同维度
        self.depth_proj = nn.Linear(depth_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        # 可学习的位置编码 (2个token: depth, proprio)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Linear(d_model * 2, d_model)  # 拼接两个token后投影

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        depth_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播

        Args:
            depth_features: [batch, depth_dim] 深度特征
            proprio_features: [batch, proprio_dim] 本体感受特征

        Returns:
            fused_features: [batch, d_model] 融合特征
        """
        batch_size = depth_features.shape[0]

        # 投影到相同维度
        depth_token = self.depth_proj(depth_features).unsqueeze(1)  # [batch, 1, d_model]
        proprio_token = self.proprio_proj(proprio_features).unsqueeze(1)  # [batch, 1, d_model]

        # 拼接成序列 [batch, 2, d_model]
        tokens = torch.cat([depth_token, proprio_token], dim=1)

        # 添加位置编码
        tokens = tokens + self.pos_embedding

        # Transformer编码
        encoded = self.transformer(tokens)  # [batch, 2, d_model]

        # 拼接两个token并投影
        fused = encoded.reshape(batch_size, -1)  # [batch, 2*d_model]
        fused = self.output_proj(fused)  # [batch, d_model]

        return fused


class MemoryGRU(nn.Module):
    """GRU记忆模块

    维护时序记忆，生成包含历史信息的特征。
    输入: [batch, input_dim] = [batch, 64]
    输出: [batch, hidden_dim] = [batch, 512]

    PIE使用GRU来记忆机器人与地形的交互历史
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 隐状态，在reset时需要清零
        self.hidden_state: Optional[torch.Tensor] = None

        # 优化GRU参数存储
        self.gru.flatten_parameters()

    def reset_hidden_state(self, batch_size: int, device: torch.device):
        """重置隐状态

        Args:
            batch_size: 批次大小
            device: 设备
        """
        self.hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device
        )

    def reset_hidden_state_for_envs(self, env_ids: torch.Tensor):
        """为特定环境重置隐状态

        Args:
            env_ids: 需要重置的环境索引
        """
        if self.hidden_state is not None:
            self.hidden_state[:, env_ids, :] = 0.0

    def detach_hidden_state(self):
        """分离隐状态的计算图，防止梯度回传过长"""
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: [batch, input_dim] 输入特征

        Returns:
            output: [batch, hidden_dim] GRU输出
        """
        batch_size = x.shape[0]

        # 初始化隐状态
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.reset_hidden_state(batch_size, x.device)

        # GRU前向传播 (添加时间维度)
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        output, self.hidden_state = self.gru(x, self.hidden_state)

        return output.squeeze(1)  # [batch, hidden_dim]


class VAEHead(nn.Module):
    """VAE编码头

    实现VAE的编码器部分，输出均值和对数方差。
    用于PIE中的隐式状态表示学习。

    输入: [batch, input_dim] = [batch, 512]
    输出: mu [batch, latent_dim], logvar [batch, latent_dim]
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 共享编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
        )

        # 均值和对数方差分支
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: [batch, input_dim] 输入特征

        Returns:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """重参数化技巧

        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差
            training: 是否在训练模式

        Returns:
            z: [batch, latent_dim] 采样的隐变量
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # 推理时直接使用均值

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """计算KL散度 D_KL(q(z|x) || N(0,1))

        Args:
            mu: [batch, latent_dim] 均值
            logvar: [batch, latent_dim] 对数方差

        Returns:
            kl: [batch] 每个样本的KL散度
        """
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl


class MultiHeadDecoder(nn.Module):
    """多头解码器

    从GRU输出解码多个估计目标：
    1. v_hat: 基座速度估计 (3D)
    2. h_f_hat: 足部间隙估计 (4D)
    3. z_m: 高度图编码 (latent_dim)
    4. z_t: VAE隐变量 (latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 512,
        velocity_dim: int = 3,
        foot_clearance_dim: int = 4,
        latent_dim: int = 32,
        height_map_dim: int = 187,  # 高度扫描点数 (Isaac Lab height_scanner默认)
        proprio_dim: int = 57,  # 本体感受维度（用于状态预测）
    ):
        super().__init__()

        # 速度估计头 (显式物理量)
        self.velocity_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, velocity_dim),
        )

        # 足部间隙估计头 (显式物理量)
        self.foot_clearance_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, foot_clearance_dim),
        )

        # 高度图编码头 (编码潜在向量)
        self.height_map_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

        # 高度图解码器 (用于训练时重建监督)
        self.height_map_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, height_map_dim),
        )

        # VAE头 (用于隐式状态预测)
        self.vae_head = VAEHead(input_dim, latent_dim)

        # 状态解码器 (重建下一步本体感受)
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, proprio_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        gru_output: torch.Tensor,
        training: bool = True,
    ) -> dict:
        """前向传播

        Args:
            gru_output: [batch, input_dim] GRU输出特征
            training: 是否在训练模式

        Returns:
            outputs: 包含所有估计结果的字典
        """
        # 显式物理量估计
        v_hat = self.velocity_head(gru_output)  # [batch, 3]
        h_f_hat = self.foot_clearance_head(gru_output)  # [batch, 4]

        # 高度图编码和解码
        z_m = self.height_map_encoder(gru_output)  # [batch, latent_dim]
        m_hat = self.height_map_decoder(z_m)  # [batch, height_map_dim]

        # VAE隐式状态
        mu, logvar = self.vae_head(gru_output)
        z_t = self.vae_head.reparameterize(mu, logvar, training)  # [batch, latent_dim]

        # 状态预测 (重建下一步本体感受)
        o_next_hat = self.state_decoder(z_t)  # [batch, proprio_dim]

        return {
            "v_hat": v_hat,              # 速度估计
            "h_f_hat": h_f_hat,          # 足部间隙估计
            "z_m": z_m,                  # 高度图编码
            "m_hat": m_hat,              # 高度图重建
            "z_t": z_t,                  # VAE隐变量
            "mu": mu,                    # VAE均值
            "logvar": logvar,            # VAE对数方差
            "o_next_hat": o_next_hat,    # 下一步状态预测
        }
