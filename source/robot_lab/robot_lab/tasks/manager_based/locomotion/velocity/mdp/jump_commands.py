# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""跳跃命令控制器 - 用于 Helios 双足轮腿机器人垂直跳跃训练

简化状态机流程（参考 GO2_Spring_Jump）：
    待机 (jump_cmd=0) → 跳跃 (jump_cmd=1) → 腾空 (was_in_flight) → 落地 (has_jumped)

jump_cmd 含义：
    - jump_cmd = 0: 待机/站立
    - jump_cmd = 1: 跳跃指令已触发
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
    """跳跃命令控制器 - 简化版状态机

    命令格式: [lin_vel_x, jump_cmd, ang_vel_z]
    - lin_vel_x: 前向目标速度（原地跳跃时为0）
    - jump_cmd: 跳跃指令 (0=待机, 1=跳跃)
    - ang_vel_z: 偏航角速度（原地跳跃时为0）

    状态变量:
    - was_in_flight: 是否曾经腾空（双脚离地）
    - has_jumped: 是否完成跳跃（腾空后落地）
    """

    cfg: "JumpCommandCfg"

    def __init__(self, cfg: "JumpCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # 获取机器人资产
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 获取脚部刚体索引
        self._resolve_feet_body_ids()

        # === 跳跃状态追踪张量 ===
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
        # 上一帧的接触状态
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_body_ids), dtype=torch.bool, device=self.device)
        # 最大高度记录
        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # === 命令缓冲区 ===
        # [lin_vel_x, jump_cmd, ang_vel_z]
        self._raw_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # === 跳跃成功率指标 ===
        self.metrics = {
            "jump_takeoff_rate": torch.zeros(self.num_envs, device=self.device),
            "jump_success_rate": torch.zeros(self.num_envs, device=self.device),
            "avg_max_height": torch.zeros(self.num_envs, device=self.device),
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
            f"\t脚部刚体: {self.feet_body_ids.tolist()}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """返回当前命令缓冲区 [num_envs, 3]"""
        return self._raw_command

    def _update_metrics(self):
        """更新跳跃成功率指标"""
        self.metrics["jump_takeoff_rate"] = self.was_in_flight.float()
        self.metrics["jump_success_rate"] = self.has_jumped.float()
        self.metrics["avg_max_height"] = self.max_height

        # 打印统计
        takeoff_rate = self.was_in_flight.float().mean().item() * 100
        land_rate = self.has_jumped.float().mean().item() * 100
        avg_height = self.max_height.mean().item()
        # print(
        #     f"[Jump] 起跳率: {takeoff_rate:.1f}% | "
        #     f"落地率: {land_rate:.1f}% | "
        #     f"平均高度: {avg_height:.3f}m"
        # )

    def _resample_command(self, env_ids: Sequence[int]):
        """重采样命令并重置跳跃状态"""
        if len(env_ids) == 0:
            return

        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 重置跳跃状态
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False

        # 记录初始位置
        self.init_poses[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.landing_poses[env_ids] = self.init_poses[env_ids].clone()

        # 重置最大高度
        self.max_height[env_ids] = 0.0

        # 重置接触状态
        self.last_contacts[env_ids] = False

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
        """更新跳跃状态机（简化版）

        状态机逻辑：
        1. 到达触发帧 → jump_cmd = 1
        2. 双脚离地 → was_in_flight = True
        3. 腾空后有接触 → has_jumped = True
        """
        # 获取接触传感器数据
        contact_sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]

        # === 接触检测 ===
        feet_contact_forces = contact_sensor.data.net_forces_w[:, self.feet_body_ids, 2]
        feet_contact = feet_contact_forces > self.cfg.contact_threshold
        # 滤波：当前帧或上一帧有接触都算接触
        contact_filt = torch.logical_or(feet_contact, self.last_contacts)
        self.last_contacts = feet_contact.clone()

        current_frame = self._env.episode_length_buf

        # === 1. 触发跳跃命令 ===
        # 到达触发帧，设置 jump_cmd = 1
        should_trigger = (current_frame >= self.jump_trigger_frame) & (self._raw_command[:, 1] == 0.0)
        self._raw_command[should_trigger, 1] = 1.0

        # === 2. 检测腾空 ===
        # 双脚都离地 + 跳跃命令已触发
        all_feet_in_air = ~torch.any(contact_filt, dim=1)
        just_took_off = all_feet_in_air & (self._raw_command[:, 1] > 0) & ~self.was_in_flight
        self.was_in_flight = torch.logical_or(self.was_in_flight, just_took_off)

        # === 3. 检测落地 ===
        # 曾经腾空 + 现在有接触
        any_contact = torch.any(contact_filt, dim=1)
        just_landed = any_contact & self.was_in_flight & ~self.has_jumped

        if just_landed.any():
            self.landing_poses[just_landed] = self.robot.data.root_pos_w[just_landed, :2]

        self.has_jumped = torch.logical_or(self.has_jumped, just_landed)

        # === 4. 更新最大高度 ===
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

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()
