# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""跳跃命令控制器 - 用于 Helios 双足轮腿机器人向前跳跃训练

状态机流程：
    向前运动 (jump_cmd=0) → 蓄力下蹲 (0<jump_cmd<1, 渐变) → 起跳爆发 (jump_cmd=1) → 腾空 → 落地

jump_cmd 含义：
    - jump_cmd = 0: 正常向前运动
    - 0 < jump_cmd < 1: 蓄力阶段（从0渐变到1，表示蓄力进度）
    - jump_cmd = 1: 起跳爆发时刻
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JumpCommand(CommandTerm):
    """跳跃命令控制器 - 管理跳跃状态机

    命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    - lin_vel_x: 前向目标速度
    - jump_cmd: 跳跃指令 (0=运动, 0~1=蓄力进度, 1=起跳爆发)
    - ang_vel_z: 偏航角速度

    内部状态:
    - is_charging: 是否处于蓄力阶段
    - is_launching: 是否处于起跳爆发阶段
    - was_in_flight: 是否曾经腾空
    - has_jumped: 是否完成跳跃
    """

    cfg: "JumpCommandCfg"

    def __init__(self, cfg: "JumpCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # 获取机器人资产
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 获取脚部刚体索引
        self._resolve_feet_body_ids()

        # === 跳跃状态追踪张量 ===
        # 是否处于蓄力阶段
        self.is_charging = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 是否处于起跳爆发阶段
        self.is_launching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 是否曾经腾空
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 是否完成跳跃
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 落地时的 XY 位置
        self.landing_poses = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        # episode 开始时的 XY 位置
        self.init_poses = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        # 跳跃触发帧（随机）
        self.jump_trigger_frame = torch.randint(
            self.cfg.jump_trigger_range[0],
            self.cfg.jump_trigger_range[1],
            (self.num_envs,),
            device=self.device,
        )
        # 蓄力开始帧
        self.charge_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 蓄力持续帧数
        self.charge_duration_frames = int(self.cfg.charge_duration / self._env.step_dt)
        # 上一帧的接触状态
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_body_ids), dtype=torch.bool, device=self.device)
        # 最大高度记录
        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # === 命令缓冲区 ===
        # [lin_vel_x, jump_cmd, ang_vel_z]
        self._raw_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # === 跳跃成功率指标 ===
        # 注意：metrics 值必须是 torch.Tensor，Isaac Lab 会对其进行索引操作
        self.metrics = {
            "jump_takeoff_rate": torch.zeros(self.num_envs, device=self.device),  # 起跳成功率
            "jump_success_rate": torch.zeros(self.num_envs, device=self.device),  # 落地成功率
            "avg_max_height": torch.zeros(self.num_envs, device=self.device),     # 最大高度
        }

    def _resolve_feet_body_ids(self):
        """解析脚部刚体索引"""
        feet_names = self.cfg.feet_body_names
        self.feet_body_ids = []
        for name in feet_names:
            body_ids = self.robot.find_bodies(name)[0]
            if len(body_ids) > 0:
                self.feet_body_ids.extend(body_ids)

        if len(self.feet_body_ids) == 0:
            raise ValueError(f"未找到脚部刚体: {feet_names}")

        self.feet_body_ids = torch.tensor(self.feet_body_ids, device=self.device, dtype=torch.long)

    def __str__(self) -> str:
        return (
            "JumpCommand:\n"
            f"\t环境数量: {self.num_envs}\n"
            f"\t跳跃触发帧范围: {self.cfg.jump_trigger_range}\n"
            f"\t蓄力时间: {self.cfg.charge_duration}s ({self.charge_duration_frames} frames)\n"
            f"\t前向速度范围: {self.cfg.ranges.lin_vel_x}\n"
            f"\t脚部刚体: {self.feet_body_ids.tolist()}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """返回当前命令缓冲区 [num_envs, 3]"""
        return self._raw_command

    def _update_metrics(self):
        """更新跳跃成功率指标

        指标说明：
        - jump_takeoff_rate: 是否成功腾空（每个环境 0/1）
        - jump_success_rate: 是否成功落地（每个环境 0/1）
        - avg_max_height: 当前最大高度（每个环境）
        """
        # 更新指标张量（每个环境的状态）
        self.metrics["jump_takeoff_rate"] = self.was_in_flight.float()
        self.metrics["jump_success_rate"] = self.has_jumped.float()
        self.metrics["avg_max_height"] = self.max_height

        #打印已禁用
        takeoff_rate = self.was_in_flight.float().mean().item() * 100
        land_rate = self.has_jumped.float().mean().item() * 100
        avg_height = self.max_height.mean().item()
        print(
            f"[Jump] 起跳率: {takeoff_rate:.1f}% | "
            f"落地率: {land_rate:.1f}% | "
            f"平均高度: {avg_height:.3f}m"
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """重采样命令并重置跳跃状态"""
        if len(env_ids) == 0:
            return

        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 重置跳跃状态
        self.is_charging[env_ids] = False
        self.is_launching[env_ids] = False
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False

        # 记录初始位置
        self.init_poses[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.landing_poses[env_ids] = self.init_poses[env_ids].clone()

        # 重置最大高度
        self.max_height[env_ids] = 0.0

        # 重置接触状态
        self.last_contacts[env_ids] = False

        # 重置蓄力开始帧
        self.charge_start_frame[env_ids] = 0

        # 重新随机跳跃触发帧
        self.jump_trigger_frame[env_ids] = torch.randint(
            self.cfg.jump_trigger_range[0],
            self.cfg.jump_trigger_range[1],
            (len(env_ids),),
            device=self.device,
        )

        # 采样前向速度和角速度命令
        r = self.cfg.ranges
        self._raw_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            r.lin_vel_x[0], r.lin_vel_x[1]
        )
        # jump_cmd 初始为 0
        self._raw_command[env_ids, 1] = 0.0
        # 角速度
        self._raw_command[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            r.ang_vel_z[0], r.ang_vel_z[1]
        )

    def _update_command(self):
        """更新跳跃状态机

        jump_cmd 逻辑：
        - 0: 正常运动
        - 0→1 渐变: 蓄力阶段（下蹲压低重心）
        - 1: 起跳爆发时刻
        """
        # 获取接触传感器数据
        contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]

        # === 腾空检测：只检测脚部 ===
        feet_contact_forces = contact_sensor.data.net_forces_w[:, self.feet_body_ids, 2]
        feet_contact = feet_contact_forces > self.cfg.contact_threshold
        feet_contact_filt = torch.logical_or(feet_contact, self.last_contacts)
        self.last_contacts = feet_contact.clone()

        # === 落地检测：检测任意身体部位 ===
        all_contact_forces = contact_sensor.data.net_forces_w[:, :, 2]  # 所有刚体 Z 方向接触力
        any_body_contact = torch.any(all_contact_forces > self.cfg.contact_threshold, dim=1)

        current_frame = self._env.episode_length_buf

        # === 状态机逻辑 ===

        # 1. 检测是否该进入蓄力阶段（首次触发）
        should_start_charge = (
            (current_frame >= self.jump_trigger_frame)
            & ~self.is_charging
            & ~self.is_launching
            & ~self.has_jumped
        )
        if should_start_charge.any():
            self.is_charging[should_start_charge] = True
            self.charge_start_frame[should_start_charge] = current_frame[should_start_charge]

        # 2. 更新蓄力阶段的 jump_cmd（0→1 渐变）
        charging_mask = self.is_charging
        if charging_mask.any():
            charge_elapsed = current_frame[charging_mask] - self.charge_start_frame[charging_mask]
            # 蓄力进度：从 0 渐变到 1（但不超过 1）
            charge_progress = charge_elapsed.float() / self.charge_duration_frames
            charge_progress = torch.clamp(charge_progress, 0.0, 0.99)  # 蓄力阶段不到 1
            self._raw_command[charging_mask, 1] = charge_progress

        # 3. 检测蓄力完成 → 进入起跳爆发
        charge_elapsed_all = current_frame - self.charge_start_frame
        should_launch = self.is_charging & (charge_elapsed_all >= self.charge_duration_frames)
        if should_launch.any():
            self.is_charging[should_launch] = False
            self.is_launching[should_launch] = True
            self._raw_command[should_launch, 1] = 1.0  # 起跳爆发：jump_cmd = 1

       
        
        # 4. 检测腾空状态（起跳后双脚离地）
        all_feet_in_air = ~torch.any(feet_contact_filt, dim=1)
        just_took_off = all_feet_in_air & self.is_launching & ~self.was_in_flight
        # 腾空后关闭起跳状态（起跳阶段结束）
        if just_took_off.any():
            self.is_launching[just_took_off] = False
        self.was_in_flight = torch.logical_or(self.was_in_flight, just_took_off)

        # 5. 检测落地（任意身体部位着地）
        just_landed = any_body_contact & self.was_in_flight & ~self.has_jumped

        if just_landed.any():
            self.landing_poses[just_landed] = self.robot.data.root_pos_w[just_landed, :2]
            self.is_launching[just_landed] = False
            self._raw_command[just_landed, 1] =0

        self.has_jumped = torch.logical_or(self.has_jumped, just_landed)
     
     
        # 6. 更新最大高度
        current_height = self.robot.data.root_pos_w[:, 2]
        self.max_height = torch.maximum(self.max_height, current_height)


@configclass
class JumpCommandCfg(CommandTermCfg):
    """跳跃命令控制器配置"""

    class_type: type = JumpCommand

    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    feet_body_names: list[str] = [".*_foot_link"]
    contact_threshold: float = 1.0
    jump_trigger_range: tuple[int, int] = (50, 60)
    charge_duration: float = 0.5  # 蓄力持续时间（秒）

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (0.5, 1.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()
