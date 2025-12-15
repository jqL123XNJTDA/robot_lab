# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel - 使用指数移动平均(EMA)跟踪奖励

    解决episode边界时序问题：当episode长度(~904步)与检查间隔(1000步)不对齐时，
    原逻辑使用episode_sums会因刚reset而得到极低值。
    EMA方案独立于episode边界，每步更新滑动平均。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    # 初始化 (仅在第一步)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

        # 初始化EMA追踪器，alpha=0.001对应约1000步有效窗口
        env._lin_vel_reward_ema = torch.zeros(1, device=env.device)

    # 每步更新EMA（使用当前步的即时奖励）
    if hasattr(env.reward_manager, '_term_sums') and reward_term_name in env.reward_manager._term_sums:
        current_reward = torch.mean(env.reward_manager._term_sums[reward_term_name][env_ids])
        alpha = 0.001
        env._lin_vel_reward_ema = alpha * current_reward + (1 - alpha) * env._lin_vel_reward_ema

    # 每 max_episode_length 步检查一次课程升级
    if env.common_step_counter % env.max_episode_length == 0 and env.common_step_counter > 0:
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # 使用EMA判断是否升级，阈值为最大奖励的80%
        if env._lin_vel_reward_ema.item() > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel - 使用指数移动平均(EMA)跟踪奖励

    解决episode边界时序问题，与command_levels_lin_vel同理。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    # 初始化 (仅在第一步)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

        # 初始化EMA追踪器
        env._ang_vel_reward_ema = torch.zeros(1, device=env.device)

    # 每步更新EMA
    if hasattr(env.reward_manager, '_term_sums') and reward_term_name in env.reward_manager._term_sums:
        current_reward = torch.mean(env.reward_manager._term_sums[reward_term_name][env_ids])
        alpha = 0.001
        env._ang_vel_reward_ema = alpha * current_reward + (1 - alpha) * env._ang_vel_reward_ema

    # 每 max_episode_length 步检查一次课程升级
    if env.common_step_counter % env.max_episode_length == 0 and env.common_step_counter > 0:
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # 使用EMA判断是否升级
        if env._ang_vel_reward_ema.item() > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
