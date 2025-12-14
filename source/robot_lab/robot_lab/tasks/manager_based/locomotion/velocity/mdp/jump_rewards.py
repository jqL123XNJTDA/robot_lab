# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""跳跃专用奖励函数 - 用于 Helios 双足轮腿机器人垂直跳跃训练

简化状态机（参考 GO2_Spring_Jump）：
    待机 (jump_cmd=0) → 跳跃触发 (jump_cmd=1) → 腾空 (was_in_flight) → 落地 (has_jumped)

奖励函数列表：
1. jump_vertical_velocity - 腾空时 Z 轴速度奖励
2. jump_flight_reward - 腾空奖励
3. jump_base_height_flight - 腾空高度奖励
4. jump_forward_velocity - 腾空时前向速度跟踪
5. jump_wheel_lock - 轮子锁定惩罚（跳跃时）
6. jump_landing_stability - 落地稳定性奖励

分阶段奖励：
- jump_pre_charge_height - 待机阶段：保持目标高度
- jump_charge_crouch - 起跳准备阶段：压低重心（jump_cmd=1 且尚未腾空）
- jump_charge_feet_contact - 起跳准备阶段：双脚着地
- jump_launch_upward - 起跳阶段：向上速度奖励
- jump_flight_height - 腾空阶段：高度奖励
- jump_land_stable - 落地阶段：稳定性奖励
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .jump_commands import JumpCommand


# =============================================================================
# 分阶段奖励函数（根据 jump_cmd 状态给予不同奖励）
# =============================================================================


def jump_pre_charge_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """运动阶段：保持目标高度惩罚

    在蓄力指令之前 (jump_cmd=0)，惩罚偏离目标高度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height: 运动时目标高度 [m]
        asset_cfg: 机器人资产配置

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # 只在运动阶段生效：jump_cmd == 0
    jump_cmd_val = cmd[:, 1]  # 命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    pre_charge = jump_cmd_val == 0.0

    if not pre_charge.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 高度偏差惩罚
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(current_height - target_height)
    penalty = height_error * pre_charge.float()

    return penalty


def jump_pre_charge_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """运动阶段：X 速度追踪奖励

    在蓄力指令之前，奖励跟踪目标前向速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 只在运动阶段（蓄力前）生效
    jump_cmd_val = cmd[:, 1]  # 命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    pre_charge = jump_cmd_val == 0.0

    if not pre_charge.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取基座坐标系下的前向速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # X 轴速度追踪（指数核）
    target_vel_x = cmd[:, 0]  # lin_vel_x
    vel_error = torch.square(target_vel_x - root_lin_vel_b[:, 0])
    reward = torch.exp(-vel_error / 0.25) * pre_charge.float()

    return reward


def jump_charge_crouch(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起跳准备阶段：压低重心奖励（简化版）

    在跳跃指令触发后（jump_cmd=1）、腾空前，奖励机器人压低重心蓄力

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height: 蓄力时目标高度 [m]
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空 → 起跳准备阶段
    jump_triggered = cmd[:, 1] == 1.0
    is_preparing = jump_triggered & ~jump_cmd.was_in_flight

    if not is_preparing.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 高度奖励：越接近目标低姿态越好
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(current_height - target_height)
    reward = torch.exp(-height_error * 10.0) * is_preparing.float()

    return reward


def jump_charge_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起跳准备阶段：X 速度追踪奖励（简化版）

    在跳跃指令触发后、腾空前，奖励保持目标前向速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空
    jump_triggered = cmd[:, 1] == 1.0
    is_preparing = jump_triggered & ~jump_cmd.was_in_flight

    if not is_preparing.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取基座坐标系下的前向速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # X 轴速度追踪（指数核）
    target_vel_x = cmd[:, 0]  # lin_vel_x
    vel_error = torch.square(target_vel_x - root_lin_vel_b[:, 0])
    reward = torch.exp(-vel_error / 0.25) * is_preparing.float()

    return reward


def jump_charge_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """起跳准备阶段：双脚着地惩罚（简化版）

    在跳跃指令触发后、腾空前，惩罚脚离地（应该双脚着地蓄力）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        contact_threshold: 接触力阈值 [N]
        sensor_cfg: 接触传感器配置（应配置 body_names 为脚部）

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空
    jump_triggered = cmd[:, 1] == 1.0
    is_preparing = jump_triggered & ~jump_cmd.was_in_flight

    if not is_preparing.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取脚部接触力（使用 body_ids，若为 None 则使用全部刚体）
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    feet_contact_forces = contact_sensor.data.net_forces_w[:, body_ids, 2]

    # 检测每只脚是否接触地面
    feet_contact = feet_contact_forces > contact_threshold  # [num_envs, num_feet]

    # 统计离地的脚数量（接触力 < 阈值）
    feet_in_air = ~feet_contact  # True = 离地
    num_feet_in_air = torch.sum(feet_in_air.float(), dim=1)  # 离地脚数量

    # 惩罚：离地脚越多惩罚越大
    penalty = num_feet_in_air * is_preparing.float()

    return penalty


def jump_launch_upward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起跳阶段：向上速度奖励（简化版）

    在跳跃指令触发后（jump_cmd=1）、腾空前，奖励向上速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空
    jump_triggered = cmd[:, 1] == 1.0
    is_launching = jump_triggered & ~jump_cmd.was_in_flight

    if not is_launching.any():
        return torch.zeros(env.num_envs, device=env.device)

    # Z 轴速度奖励（向上为正）
    z_vel = asset.data.root_lin_vel_w[:, 2]
    reward = torch.clamp(z_vel, min=0) * is_launching.float()

    return reward


def jump_launch_vertical_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起跳阶段：垂直爆发速度奖励（简化版）

    在跳跃指令触发后、腾空前，Z 轴速度越大奖励越高（线性关系）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        velocity_scale: 速度缩放系数，调整奖励敏感度
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]（速度越大，奖励越高）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空
    jump_triggered = cmd[:, 1] == 1.0
    is_launching = jump_triggered & ~jump_cmd.was_in_flight

    if not is_launching.any():
        return torch.zeros(env.num_envs, device=env.device)

    # Z 轴速度（向上为正），速度越大奖励越高
    z_vel = asset.data.root_lin_vel_w[:, 2]
    # 只奖励正向速度，负速度不惩罚（返回0）
    reward = torch.clamp(z_vel * velocity_scale, min=0) * is_launching.float()

    return reward


def jump_launch_grf(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_force: float = 500.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """起跳阶段：地面反作用力奖励（简化版）

    在跳跃指令触发后、腾空前，脚底蹬地力度越大，奖励越高
    为防止物理引擎不稳定，力值有上限截断

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        max_force: 最大力值上限 [N]，防止无限大的力导致不稳定
        sensor_cfg: 接触传感器配置（应配置 body_names 为脚部）

    Returns:
        奖励张量 [num_envs]（力越大，奖励越高，但有上限）
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 简化逻辑：jump_cmd=1 且尚未腾空
    jump_triggered = cmd[:, 1] == 1.0
    is_launching = jump_triggered & ~jump_cmd.was_in_flight

    if not is_launching.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取脚部接触力（Z方向，即垂直于地面）
    feet_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # 只取正向力（向上的反作用力），并截断到最大值
    grf = torch.clamp(feet_contact_forces, min=0, max=max_force)

    # 对所有脚的力求和，然后归一化到 [0, 1] 范围
    total_grf = torch.sum(grf, dim=1)
    num_feet = len(sensor_cfg.body_ids) if sensor_cfg.body_ids is not None else feet_contact_forces.shape[1]
    normalized_grf = total_grf / (max_force * num_feet)

    reward = normalized_grf * is_launching.float()

    return reward


def jump_flight_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空阶段：高度奖励

    在腾空阶段 (was_in_flight = True, has_jumped = False)，奖励达到目标高度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height: 目标跳跃高度 [m]
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在腾空阶段给予奖励
    in_flight = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    if not in_flight.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 高度奖励
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(current_height - target_height)
    height_reward = torch.exp(-height_error * 5.0)

    # 重力投影 Z 分量：直立时为 -1，倾斜时偏离 -1
    # 转换为 [0, 1] 范围：越接近 -1（直立），upright_factor 越接近 1
    projected_gravity_z = asset.data.projected_gravity_b[:, 2]
    upright_factor = torch.clamp(-projected_gravity_z, min=0, max=1)

    # 综合奖励：高度 × 直立因子
    reward = height_reward * upright_factor * in_flight.float()

    return reward


def jump_flight_action_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空阶段：动作惩罚（禁止空中乱蹬腿）

    在腾空阶段，惩罚腿部关节的剧烈动作变化。
    空中乱蹬腿会通过角动量守恒反作用到基座，导致翻车。

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置（应配置 joint_names 为腿部关节）

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在腾空阶段惩罚
    in_flight = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    if not in_flight.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取腿部关节速度（关节角速度越大，惩罚越大）
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # L2 范数惩罚
    penalty = torch.sum(torch.square(joint_vel), dim=1) * in_flight.float()

    return penalty


def jump_flight_action_rate_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """腾空阶段：动作变化率惩罚（禁止空中动作剧烈变化）

    在腾空阶段，惩罚动作的剧烈变化（当前动作与上一步动作的差异）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在腾空阶段惩罚
    in_flight = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    if not in_flight.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 动作变化率惩罚
    action_diff = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(torch.square(action_diff), dim=1) * in_flight.float()

    return penalty


def jump_flight_foot_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height_below: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: list[str] = ["left_foot_link", "right_foot_link"],
) -> torch.Tensor:
    """腾空阶段：脚部高度惩罚（防止空中乱蹬腿）

    在腾空阶段，惩罚脚部高度偏离目标位置（base 高度 - target_height_below）。
    目标是让脚保持在机体下方 target_height_below 米处，过高或过低都惩罚。

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height_below: 脚应在 base 下方的目标距离 [m]
        asset_cfg: 机器人资产配置
        foot_body_names: 脚部刚体名称列表

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在腾空阶段惩罚
    in_flight = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    if not in_flight.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取 base 高度
    base_height = asset.data.root_pos_w[:, 2]  # [num_envs]

    # 目标脚高度 = base 高度 - target_height_below（脚在机体下方）
    target_foot_height = base_height - target_height_below  # [num_envs]

    # 获取脚部刚体高度
    foot_ids = asset.find_bodies(foot_body_names)[0]
    foot_heights = asset.data.body_pos_w[:, foot_ids, 2]  # [num_envs, num_feet]

    # 脚部高度偏离目标的误差（过高或过低都惩罚）
    height_error = foot_heights - target_foot_height.unsqueeze(1)  # [num_envs, num_feet]

    # 惩罚：误差的平方和（过高或过低都惩罚）
    penalty = torch.sum(torch.square(height_error), dim=1) * in_flight.float()

    return penalty


def jump_land_stable(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地阶段：稳定性奖励

    在落地后 (has_jumped = True)，奖励稳定姿态和低速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在已落地后给予奖励
    has_jumped = jump_cmd.has_jumped

    if not has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 姿态稳定奖励
    projected_gravity = asset.data.projected_gravity_b
    orientation_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    # 角速度惩罚
    root_ang_vel = asset.data.root_ang_vel_w
    ang_vel_penalty = torch.sum(torch.square(root_ang_vel), dim=1)

    reward = torch.exp(-orientation_error * 10.0 - ang_vel_penalty * 0.1) * has_jumped.float()

    return reward


def jump_land_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地阶段：恢复目标高度惩罚

    在落地后 (has_jumped = True)，惩罚偏离目标高度，促使机器人恢复正常站立姿态

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height: 落地后目标高度 [m]
        asset_cfg: 机器人资产配置

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在落地后生效
    has_jumped = jump_cmd.has_jumped

    if not has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 高度偏差惩罚
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(current_height - target_height)
    penalty = height_error * has_jumped.float()

    return penalty


def jump_land_vertical_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地阶段：垂直速度阻尼惩罚（软着陆）

    在落地后，惩罚 Z 轴线速度，促使 Vz 尽快归零，消除垂直震荡

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在落地后生效
    has_jumped = jump_cmd.has_jumped

    if not has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # Z 轴速度惩罚（无论正负都惩罚）
    z_vel = asset.data.root_lin_vel_w[:, 2]
    penalty = torch.square(z_vel) * has_jumped.float()

    return penalty


def jump_land_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地阶段：X 速度追踪奖励

    在落地后，奖励恢复目标前向速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 只在落地后生效
    has_jumped = jump_cmd.has_jumped

    if not has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取基座坐标系下的前向速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # X 轴速度追踪（指数核）
    target_vel_x = cmd[:, 0]  # lin_vel_x
    vel_error = torch.square(target_vel_x - root_lin_vel_b[:, 0])
    reward = torch.exp(-vel_error / 0.25) * has_jumped.float()

    return reward


def jump_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """全阶段 Z 轴角速度惩罚（期望角速度为0）

    惩罚任何非零的偏航角速度，保持机器人不旋转

    Args:
        env: 环境实例
        command_name: 跳跃命令名称（未使用，保留接口兼容）
        std: 未使用，保留接口兼容
        asset_cfg: 机器人资产配置

    Returns:
        角速度平方 [num_envs]，配合负权重使用
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Z 轴角速度平方惩罚（期望为0）
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    penalty = torch.square(ang_vel_z)

    return penalty


def jump_track_lin_vel_x(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """全阶段 X 轴速度追踪奖励（使用 jump_command）

    在所有阶段（待机、起跳、腾空、落地）追踪目标前向速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        std: 指数核标准差
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # 获取基座坐标系下的前向速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # X 轴速度追踪（指数核）
    # 命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    target_vel_x = cmd[:, 0]  # lin_vel_x
    vel_error = torch.square(target_vel_x - root_lin_vel_b[:, 0])
    reward = torch.exp(-vel_error / std)

    return reward


def jump_phase_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    prepare_target_height: float = 0.25,
    flight_target_height: float = 0.5,
    prepare_weight: float = 1.0,
    launch_weight: float = 2.0,
    flight_weight: float = 1.5,
    land_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """综合分阶段奖励（简化版）

    根据当前阶段自动选择对应的奖励，支持不同阶段不同权重

    简化状态机阶段：
    - 待机阶段：jump_cmd=0
    - 起跳准备阶段：jump_cmd=1 且尚未腾空
    - 腾空阶段：was_in_flight=True 且 has_jumped=False
    - 落地阶段：has_jumped=True

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        prepare_target_height: 起跳准备时目标高度
        flight_target_height: 腾空时目标高度
        prepare_weight: 起跳准备阶段权重
        launch_weight: 起跳阶段权重
        flight_weight: 腾空阶段权重
        land_weight: 落地阶段权重
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    reward = torch.zeros(env.num_envs, device=env.device)

    # === 起跳准备阶段：jump_cmd=1 且尚未腾空 ===
    jump_triggered = cmd[:, 1] == 1.0
    is_preparing = jump_triggered & ~jump_cmd.was_in_flight
    if is_preparing.any():
        current_height = asset.data.root_pos_w[:, 2]
        height_error = torch.abs(current_height - prepare_target_height)
        prepare_reward = torch.exp(-height_error * 10.0) * is_preparing.float()
        reward += prepare_weight * prepare_reward

        # 起跳速度奖励
        z_vel = asset.data.root_lin_vel_w[:, 2]
        launch_reward = torch.clamp(z_vel, min=0) * is_preparing.float()
        reward += launch_weight * launch_reward

    # === 腾空阶段 ===
    in_flight = jump_cmd.was_in_flight & ~jump_cmd.has_jumped
    if in_flight.any():
        current_height = asset.data.root_pos_w[:, 2]
        height_error = torch.abs(current_height - flight_target_height)
        flight_reward = torch.exp(-height_error * 5.0) * in_flight.float()
        reward += flight_weight * flight_reward

    # === 落地阶段 ===
    has_jumped = jump_cmd.has_jumped
    if has_jumped.any():
        projected_gravity = asset.data.projected_gravity_b
        orientation_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        land_reward = torch.exp(-orientation_error * 10.0) * has_jumped.float()
        reward += land_weight * land_reward

    return reward


def jump_flight_vel_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空阶段 Z 轴速度奖励（参考 GO2 _reward_line_z）

    在跳跃指令触发后、落地前，Z 轴正向速度越大奖励越高。
    条件：jump_cmd=1 且 has_jumped=False

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 条件：jump_cmd=1 且尚未落地
    jump_triggered = cmd[:, 1] == 1.0
    active = jump_triggered & ~jump_cmd.has_jumped

    # Z 轴速度奖励：只奖励正向速度（向上），超过 0.5 m/s 饱和
    z_vel = asset.data.root_lin_vel_w[:, 2]
    vel_reward = torch.clamp(z_vel, min=0, max=0.5)

    # 重力投影 Z 分量：直立时为 -1，倾斜时偏离 -1
    # 转换为 [0, 1] 范围：越接近 -1（直立），upright_factor 越接近 1
    projected_gravity_z = asset.data.projected_gravity_b[:, 2]
    upright_factor = torch.clamp(-projected_gravity_z, min=0, max=1)

    # 综合奖励：速度 × 直立因子
    reward = vel_reward * upright_factor * active.float()

    return reward


# =============================================================================
# 原有奖励函数
# =============================================================================


def jump_vertical_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空时 Z 轴速度奖励

    在跳跃指令后、落地前，Z 轴速度越大越好（向上跳得更高）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 获取跳跃状态
    was_in_flight = jump_cmd.was_in_flight
    has_jumped = jump_cmd.has_jumped
    jump_active = env.command_manager.get_command(command_name)[:, 1] > 0  # jump_cmd > 0

    # 只在腾空阶段（已起跳但未落地）给予奖励
    in_flight_phase = was_in_flight & ~has_jumped & jump_active

    # Z 轴速度奖励（只奖励正向速度，即向上）
    z_vel = asset.data.root_lin_vel_w[:, 2]
    reward = torch.clamp(z_vel, min=0) * in_flight_phase.float()

    return reward


def jump_flight_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """腾空奖励 - 奖励成功腾空

    只要机器人曾经腾空过（所有脚离地），就给予奖励

    Args:
        env: 环境实例
        command_name: 跳跃命令名称

    Returns:
        奖励张量 [num_envs]
    """
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 腾空状态且尚未落地时给予奖励
    reward = jump_cmd.was_in_flight.float() * (~jump_cmd.has_jumped).float()

    return reward


def jump_base_height_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空高度奖励

    在腾空阶段，奖励机器人达到目标高度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        target_height: 目标跳跃高度 [m]
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在腾空阶段给予奖励
    in_flight_phase = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    # 高度奖励（使用指数核，越接近目标高度奖励越高）
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(current_height - target_height)
    reward = torch.exp(-height_error * 5.0) * in_flight_phase.float()

    return reward


def jump_forward_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_scale: float = 1.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腾空时前向速度跟踪奖励

    在腾空阶段，奖励机器人保持目标前向速度

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        velocity_scale: 目标速度放大系数
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 只在腾空阶段给予奖励
    in_flight_phase = jump_cmd.was_in_flight & ~jump_cmd.has_jumped

    # 获取基座坐标系下的前向速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # X 轴速度跟踪（目标速度 = 命令速度 * 放大系数）
    target_vel_x = cmd[:, 0] * velocity_scale
    vel_error = torch.square(target_vel_x - root_lin_vel_b[:, 0])

    reward = torch.exp(-vel_error) * in_flight_phase.float()

    return reward


def jump_wheel_lock(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跳跃时轮子锁定惩罚

    在跳跃阶段（从跳跃指令到落地），惩罚轮子转动，防止空转浪费能量

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置（应配置 joint_names 为轮子关节）

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 在跳跃阶段（jump_cmd > 0 且尚未落地）惩罚轮子转动
    in_jump_phase = (cmd[:, 1] > 0) & ~jump_cmd.has_jumped  # cmd[:, 1] 是 jump_cmd

    # 获取轮子关节速度
    wheel_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])

    # 轮子速度惩罚
    penalty = torch.sum(wheel_vel, dim=1) * in_jump_phase.float()

    return penalty


def jump_landing_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地稳定性奖励

    落地后奖励低速度和稳定姿态

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 只在已落地后给予奖励
    if not jump_cmd.has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取基座坐标系下的速度
    root_quat = asset.data.root_quat_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)

    # 落地后的线速度惩罚（XY 平面）
    lin_vel_penalty = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)

    # 落地后的角速度惩罚
    root_ang_vel_w = asset.data.root_ang_vel_w
    root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)
    ang_vel_penalty = torch.sum(torch.square(root_ang_vel_b), dim=1)

    # 姿态稳定奖励（重力投影接近 [0, 0, -1]）
    projected_gravity = asset.data.projected_gravity_b
    # 理想情况下 projected_gravity_b = [0, 0, -1]
    # 惩罚 XY 分量偏离 0
    orientation_penalty = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    # 综合奖励
    reward = torch.exp(-orientation_penalty * 10.0) - 0.1 * lin_vel_penalty - 0.05 * ang_vel_penalty
    reward = torch.clamp(reward, min=0) * jump_cmd.has_jumped.float()

    return reward


def jump_landing_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """落地位置精度奖励

    奖励落地位置接近目标位置（初始位置 + 命令偏移）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        min_height: 触发奖励的最小跳跃高度
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量 [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 只在完成跳跃后给予奖励
    if not jump_cmd.has_jumped.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 计算目标落地位置（初始位置 + 命令速度作为位置偏移）
    target_landing = jump_cmd.init_poses + cmd[:, :2]

    # 落地位置误差
    landing_error = torch.sum(torch.abs(target_landing - jump_cmd.landing_poses), dim=1)

    # 额外条件：姿态稳定 + 达到最小高度
    projected_gravity = asset.data.projected_gravity_b
    orientation_stable = torch.sum(torch.abs(projected_gravity[:, :2]), dim=1) < 0.6
    reached_height = jump_cmd.max_height > min_height

    # 综合条件
    valid_jump = jump_cmd.has_jumped & orientation_stable & reached_height

    reward = torch.exp(-landing_error) * valid_jump.float()

    return reward


def jump_collision(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """跳跃时碰撞惩罚

    惩罚非脚部的接触（如基座、小腿等）

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        threshold: 接触力阈值
        sensor_cfg: 接触传感器配置（应配置 body_names 为非脚部刚体）

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)

    # 获取非脚部刚体的接触力
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_magnitude = torch.norm(contact_forces, dim=-1)

    # 统计超过阈值的接触数量
    collision_count = torch.sum((contact_magnitude > threshold).float(), dim=1)

    return collision_count


def jump_wrong_timing_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    jump_cmd_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """错误时机跳跃惩罚

    当 jump_cmd < threshold 时（运动阶段或蓄力初期），惩罚向上速度。
    防止策略在不该跳的时候跳。

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        jump_cmd_threshold: 跳跃命令阈值，低于此值时惩罚跳跃
        asset_cfg: 机器人资产配置

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # jump_cmd < threshold 时不应该跳跃
    jump_cmd_val = cmd[:, 1]  # 命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    should_not_jump = jump_cmd_val < jump_cmd_threshold

    # 向上速度（只惩罚正向速度）
    z_vel = asset.data.root_lin_vel_w[:, 2]
    upward_vel = torch.clamp(z_vel, min=0)

    # 惩罚：不该跳时向上速度越大惩罚越大
    penalty = upward_vel * should_not_jump.float()

    return penalty


def jump_idle_land_action_rate_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """待机和落地阶段：动作平滑惩罚

    在待机阶段（jump_cmd=0）和落地阶段（has_jumped=True），惩罚动作变化率。
    这两个阶段机器人应该保持平稳，动作不应剧烈变化。

    Args:
        env: 环境实例
        command_name: 跳跃命令名称

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 待机阶段：jump_cmd == 0
    jump_cmd_val = cmd[:, 1]
    is_idle = jump_cmd_val == 0.0

    # 落地阶段：has_jumped == True
    has_landed = jump_cmd.has_jumped

    # 在这两个阶段需要动作平滑
    should_smooth = is_idle | has_landed

    if not should_smooth.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 动作变化率惩罚（当前动作与上一步动作的差异）
    action_diff = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(torch.square(action_diff), dim=1) * should_smooth.float()

    return penalty


def jump_idle_land_feet_air_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """待机和落地阶段：双脚离地惩罚

    在待机阶段（jump_cmd=0）和落地阶段（has_jumped=True），惩罚双脚离地。
    这两个阶段机器人应该双脚稳定接触地面。

    Args:
        env: 环境实例
        command_name: 跳跃命令名称
        contact_threshold: 接触力阈值 [N]，低于此值视为离地
        sensor_cfg: 接触传感器配置（应配置 body_names 为脚部）

    Returns:
        惩罚张量 [num_envs]（正值，配合负权重使用）
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd: JumpCommand = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)

    # 待机阶段：jump_cmd == 0
    jump_cmd_val = cmd[:, 1]
    is_idle = jump_cmd_val == 0.0

    # 落地阶段：has_jumped == True
    has_landed = jump_cmd.has_jumped

    # 在这两个阶段需要双脚着地
    should_contact = is_idle | has_landed

    if not should_contact.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 获取脚部接触力（Z 方向）
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    feet_contact_forces = contact_sensor.data.net_forces_w[:, body_ids, 2]

    # 检测每只脚是否接触地面
    feet_contact = feet_contact_forces > contact_threshold  # [num_envs, num_feet]

    # 统计离地的脚数量（接触力 < 阈值）
    feet_in_air = ~feet_contact  # True = 离地
    num_feet_in_air = torch.sum(feet_in_air.float(), dim=1)  # 离地脚数量

    # 惩罚：离地脚越多惩罚越大
    penalty = num_feet_in_air * should_contact.float()

    return penalty
