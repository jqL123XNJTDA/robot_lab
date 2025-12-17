# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) PPO配置
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024

"""
MyDog PIE PPO 训练配置。

PIE是单阶段端到端训练框架，并发优化：
1. Actor-Critic (PPO)
2. Estimator (MSE + KL损失)

配置要点：
- 使用 PIEOnPolicyRunner
- 观测历史: H2=10帧本体感受
- 深度图时序: H1=2帧
- Estimator学习率独立配置
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from isaaclab.utils import configclass


@configclass
class PIEEstimatorCfg:
    """PIE Estimator配置

    双层隐式-显式估计机制配置：
    - Level 1: 隐式(预测o_{t+1}) vs 显式(重建高度图m_t)
    - Level 2: 隐变量(z_m, z_t) vs 显式物理量(v_hat, h_f_hat)
    """

    # 本体感受配置
    proprio_dim: int = 57  # MyDog本体感受维度
    num_proprio_frames: int = 10  # H2=10帧历史

    # 深度图配置
    num_depth_frames: int = 2  # H1=2帧时序深度图
    depth_height: int = 58  # 处理后深度图高度
    depth_width: int = 87  # 处理后深度图宽度

    # 网络配置
    latent_dim: int = 32  # 隐变量维度
    height_map_dim: int = 187  # 高度图扫描点数
    foot_clearance_dim: int = 4  # 足部间隙维度 (4个轮子)

    # 编码器配置
    depth_encoder_channels: Tuple[int, ...] = (32, 64, 128)  # CNN通道数
    depth_encoder_output_dim: int = 32  # 深度编码输出维度
    proprio_encoder_hidden_dim: int = 128  # 本体感受编码器隐藏层
    proprio_encoder_output_dim: int = 64  # 本体感受编码输出维度

    # Transformer配置
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256

    # GRU配置
    gru_hidden_dim: int = 512

    # 训练配置
    learning_rate: float = 1e-3
    kl_weight: float = 0.01  # λ_kl
    state_weight: float = 1.0  # λ_state
    map_weight: float = 1.0  # λ_map
    velocity_weight: float = 1.0  # λ_vel
    foot_clearance_weight: float = 1.0  # λ_fc


@configclass
class PIEPolicyCfg:
    """PIE Actor-Critic网络配置"""

    # Actor配置
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    init_noise_std: float = 1.0

    # Critic配置
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_obs_dim: int = 247  # 特权观测维度: 60(proprio+base_lin_vel) + 187(height_scan)

    # 通用配置
    activation: str = "elu"


@configclass
class PIEAlgorithmCfg:
    """PIE PPO算法配置"""

    # PPO配置
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 8
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class PIERunnerCfg:
    """PIE训练Runner配置

    核心配置用于 PIEOnPolicyRunner
    """

    # Runner类型
    class_name: str = "PIEOnPolicyRunner"

    # 训练配置
    num_steps_per_env: int = 24  # 每环境步数
    max_iterations: int = 30000  # 最大迭代次数
    save_interval: int = 100  # 保存间隔

    # 实验配置
    experiment_name: str = "mydog_pie"
    run_name: str = ""
    logger: str = "tensorboard"

    # 设备配置
    device: str = "cuda"

    # 动作裁剪
    clip_actions: float = 100.0

    # 子配置
    estimator: PIEEstimatorCfg = PIEEstimatorCfg()
    policy: PIEPolicyCfg = PIEPolicyCfg()
    algorithm: PIEAlgorithmCfg = PIEAlgorithmCfg()

    # 检查点配置
    resume: bool = False
    load_run: str = ""
    load_checkpoint: str = "model_.*.pt"

    def to_dict(self) -> dict:
        """转换为字典格式（用于Runner）"""
        return {
            "class_name": self.class_name,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "device": self.device,
            "clip_actions": self.clip_actions,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            # Estimator配置
            "estimator": {
                "proprio_dim": self.estimator.proprio_dim,
                "num_proprio_frames": self.estimator.num_proprio_frames,
                "num_depth_frames": self.estimator.num_depth_frames,
                "depth_height": self.estimator.depth_height,
                "depth_width": self.estimator.depth_width,
                "latent_dim": self.estimator.latent_dim,
                "height_map_dim": self.estimator.height_map_dim,
                "foot_clearance_dim": self.estimator.foot_clearance_dim,
                "learning_rate": self.estimator.learning_rate,
                "kl_weight": self.estimator.kl_weight,
                "state_weight": self.estimator.state_weight,
                "map_weight": self.estimator.map_weight,
                "velocity_weight": self.estimator.velocity_weight,
                "foot_clearance_weight": self.estimator.foot_clearance_weight,
            },
            # Policy配置
            "policy": {
                "actor_hidden_dims": list(self.policy.actor_hidden_dims),
                "critic_hidden_dims": list(self.policy.critic_hidden_dims),
                "critic_obs_dim": self.policy.critic_obs_dim,
                "init_noise_std": self.policy.init_noise_std,
                "activation": self.policy.activation,
            },
            # Algorithm配置
            "algorithm": {
                "value_loss_coef": self.algorithm.value_loss_coef,
                "use_clipped_value_loss": self.algorithm.use_clipped_value_loss,
                "clip_param": self.algorithm.clip_param,
                "entropy_coef": self.algorithm.entropy_coef,
                "num_learning_epochs": self.algorithm.num_learning_epochs,
                "num_mini_batches": self.algorithm.num_mini_batches,
                "learning_rate": self.algorithm.learning_rate,
                "schedule": self.algorithm.schedule,
                "gamma": self.algorithm.gamma,
                "lam": self.algorithm.lam,
                "desired_kl": self.algorithm.desired_kl,
                "max_grad_norm": self.algorithm.max_grad_norm,
            },
        }


# ==============================================================================
# 预设配置
# ==============================================================================


@configclass
class MyDogPIEFlatRunnerCfg(PIERunnerCfg):
    """MyDog PIE Flat环境Runner配置"""

    experiment_name: str = "mydog_pie_flat"

    def __post_init__(self):
        # 平地环境配置
        self.max_iterations = 20000
        # Critic不需要高度扫描，只需要基本特权观测
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
        # + joint_pos(16) + joint_vel(16) + actions(16) = 60D
        self.policy.critic_obs_dim = 60


@configclass
class MyDogPIERoughRunnerCfg(PIERunnerCfg):
    """MyDog PIE Rough环境Runner配置"""

    experiment_name: str = "mydog_pie_rough"

    def __post_init__(self):
        # 粗糙地形环境配置
        self.max_iterations = 30000
        # Critic使用高度扫描: 60 + 187(height_scan) = 247D
        self.policy.critic_obs_dim = 247


@configclass
class MyDogPIEParkourRunnerCfg(PIERunnerCfg):
    """MyDog PIE Parkour环境Runner配置"""

    experiment_name: str = "mydog_pie_parkour"

    def __post_init__(self):
        # Parkour环境配置
        self.max_iterations = 50000
        self.num_steps_per_env = 48  # 更长的rollout
        # Critic使用高度扫描: 60 + 187(height_scan) = 247D
        self.policy.critic_obs_dim = 247

        # 更保守的学习率
        self.algorithm.learning_rate = 5e-4
        self.estimator.learning_rate = 5e-4
