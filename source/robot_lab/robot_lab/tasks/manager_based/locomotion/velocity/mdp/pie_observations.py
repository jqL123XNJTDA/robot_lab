# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) 观测函数
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 实现PIE所需的观测函数：
# - 深度图像观测
# - 足部间隙观测

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    clip_range: tuple = (0.3, 3.0),
    normalize: bool = True,
) -> torch.Tensor:
    """获取深度图像观测

    从RayCasterCamera传感器获取深度图像，进行裁剪和归一化处理。

    Args:
        env: Isaac Lab环境
        sensor_cfg: 深度相机传感器配置
        clip_range: 深度裁剪范围 [min, max] (米)
        normalize: 是否归一化到 [-0.5, 0.5]

    Returns:
        depth_image: [num_envs, H, W] 处理后的深度图像
    """
    # 获取传感器
    sensor = env.scene.sensors[sensor_cfg.name]

    # 获取深度数据 (distance_to_image_plane)
    depth_data = sensor.data.output["distance_to_image_plane"]

    # 深度图形状: [num_envs, H, W, 1] -> [num_envs, H, W]
    depth = depth_data.squeeze(-1)

    # 裁剪深度范围
    min_depth, max_depth = clip_range
    depth = torch.clamp(depth, min_depth, max_depth)

    # 归一化到 [-0.5, 0.5]
    if normalize:
        depth = (depth - min_depth) / (max_depth - min_depth) - 0.5

    return depth


def depth_image_buffer(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    buffer_key: str = "depth_buffer",
    clip_range: tuple = (0.3, 3.0),
    normalize: bool = True,
) -> torch.Tensor:
    """获取时序深度图像观测（从缓冲区）

    PIE使用H1=2帧的时序深度图，此函数返回缓冲区中的历史深度图。
    缓冲区需要在Runner中维护。

    Args:
        env: Isaac Lab环境
        sensor_cfg: 深度相机传感器配置
        buffer_key: 缓冲区键名
        clip_range: 深度裁剪范围
        normalize: 是否归一化

    Returns:
        depth_buffer: [num_envs, H1, H, W] 时序深度图
    """
    # 此函数主要用于配置，实际深度缓冲区在Runner中管理
    # 返回当前帧深度图
    return depth_image(env, sensor_cfg, clip_range, normalize)


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple = ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """计算足部间隙（foot clearance）

    足部间隙是每个足部相对于其下方地形的高度差。
    这对于PIE的显式物理量估计非常重要。

    Args:
        env: Isaac Lab环境
        asset_cfg: 机器人资产配置
        foot_body_names: 足部刚体名称
        sensor_cfg: 高度扫描传感器配置

    Returns:
        foot_clearance: [num_envs, num_feet] 每个足部的离地高度
    """
    # 获取机器人资产
    asset = env.scene[asset_cfg.name]

    # 获取足部世界坐标
    # 需要先获取足部的body_ids
    foot_ids = []
    for name in foot_body_names:
        try:
            body_ids = asset.find_bodies(name)
            if len(body_ids[0]) > 0:
                foot_ids.append(body_ids[0][0])
        except Exception:
            pass

    if len(foot_ids) == 0:
        # 如果找不到足部，返回零张量
        return torch.zeros(env.num_envs, len(foot_body_names), device=env.device)

    # 获取足部位置 [num_envs, num_feet, 3]
    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]

    # 获取地形高度
    # 简化实现：使用机器人下方的地形高度作为参考
    try:
        height_sensor = env.scene.sensors[sensor_cfg.name]
        # ray_hits_w的z坐标就是地形高度
        terrain_height = height_sensor.data.ray_hits_w[:, :, 2].mean(dim=1, keepdim=True)
    except Exception:
        # 如果没有高度传感器，使用环境原点高度
        terrain_height = env.scene.env_origins[:, 2:3]

    # 计算足部间隙
    foot_clearance = feet_pos_w[:, :, 2] - terrain_height

    return foot_clearance


def foot_clearance_mydog(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """MyDog机器人的足部间隙观测

    MyDog有4个轮子作为"足部"。

    Args:
        env: Isaac Lab环境
        asset_cfg: 机器人资产配置
        sensor_cfg: 高度扫描传感器配置

    Returns:
        foot_clearance: [num_envs, 4] 4个轮子的离地高度
    """
    # MyDog的轮子刚体名称（足部）
    # 注意: 这是刚体名称，不是关节名称
    wheel_body_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

    return foot_clearance(
        env=env,
        asset_cfg=asset_cfg,
        foot_body_names=wheel_body_names,
        sensor_cfg=sensor_cfg,
    )


def proprio_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """获取本体感受观测（PIE格式）

    PIE论文中的本体感受观测 o_t 包含：
    - 基座角速度 ω_t (3D)
    - 重力投影 g_t (3D)
    - 速度命令 c_t (3D)
    - 关节角度 θ_t (12D for 四足, 16D for MyDog)
    - 关节角速度 θ̇_t (同上)
    - 上一步动作 a_{t-1} (同上)

    Args:
        env: Isaac Lab环境
        asset_cfg: 机器人资产配置

    Returns:
        proprio: [num_envs, proprio_dim] 本体感受观测
    """
    asset = env.scene[asset_cfg.name]

    # 基座角速度 [num_envs, 3]
    base_ang_vel = asset.data.root_ang_vel_b

    # 重力投影 [num_envs, 3]
    # 从四元数计算投影重力
    quat = asset.data.root_quat_w
    # 计算重力在机体坐标系下的投影
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    projected_gravity = quat_rotate_inverse(quat, gravity_w.expand(env.num_envs, -1))

    # 速度命令 [num_envs, 3]
    # 从命令管理器获取
    try:
        commands = env.command_manager.get_command("base_velocity")
        velocity_commands = commands[:, :3]  # [vx, vy, wz]
    except Exception:
        velocity_commands = torch.zeros(env.num_envs, 3, device=env.device)

    # 关节角度（相对于默认位置）
    joint_pos = asset.data.joint_pos - asset.data.default_joint_pos

    # 关节角速度
    joint_vel = asset.data.joint_vel

    # 上一步动作
    try:
        last_action = env.action_manager.action
    except Exception:
        last_action = torch.zeros_like(joint_pos)

    # 拼接所有观测
    proprio = torch.cat([
        base_ang_vel,       # 3
        projected_gravity,  # 3
        velocity_commands,  # 3
        joint_pos,          # 12 or 16
        joint_vel,          # 12 or 16
        last_action,        # 12 or 16
    ], dim=-1)

    return proprio


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """将向量从世界坐标系转换到机体坐标系

    Args:
        q: [N, 4] 四元数 (w, x, y, z)
        v: [N, 3] 世界坐标系下的向量

    Returns:
        v_body: [N, 3] 机体坐标系下的向量
    """
    # 四元数分量
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]

    # 向量叉积
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * 2.0 * q_w
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0

    return a - b + c


# 添加PIE简化奖励函数
def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """动作平滑度惩罚

    PIE论文Table I中的 smoothness 奖励：
    r = (a_t - 2*a_{t-1} + a_{t-2})^2

    Args:
        env: Isaac Lab环境

    Returns:
        smoothness_penalty: [num_envs] 平滑度惩罚
    """
    # 获取动作历史
    # 简化实现：使用action_rate的变化
    try:
        action_manager = env.action_manager
        # 计算二阶差分（加速度）
        # 这需要存储更多历史，简化为返回action_rate
        current_action = action_manager.action
        prev_action = action_manager.prev_action if hasattr(action_manager, 'prev_action') \
                      else current_action
        action_rate = current_action - prev_action
        return torch.sum(action_rate ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """碰撞惩罚

    PIE论文Table I中的 collision 奖励：
    r = -n_collision (碰撞点数)

    Args:
        env: Isaac Lab环境
        sensor_cfg: 接触力传感器配置
        threshold: 接触力阈值

    Returns:
        collision_count: [num_envs] 碰撞点数
    """
    try:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        # 获取接触力
        contact_forces = contact_sensor.data.net_forces_w
        # 计算超过阈值的接触点数
        collision_count = (contact_forces.norm(dim=-1) > threshold).sum(dim=-1).float()
        return collision_count
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
