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
) -> torch.Tensor:
    """
    修正版：基于 '平均每步得分 (Average Step Reward)' 的课程升级。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化 ---
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # --- 2. 检查逻辑 ---
    if env.common_step_counter % env.max_episode_length == 0 and env.common_step_counter > 0:

        # A. 获取分子：当前 Episode 累积奖励
        current_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]

        # B. 获取分母：当前 Episode 实际存活步数 (关键修改：不要乘 dt)
        current_steps = env.episode_length_buf[env_ids]  # [Steps]

        # C. 过滤噪音：只统计活了超过 10 步的机器人
        valid_mask = current_steps > 10

        if valid_mask.sum() > 0:
            # D. 计算：平均每步得分 (Reward per Step)
            # 修正：直接除以步数，单位变回 "Reward Value"
            avg_reward_per_step = current_sums[valid_mask] / current_steps[valid_mask]

            # E. 计算整个群体的平均表现
            mean_performance = torch.mean(avg_reward_per_step)

            # F. 获取满分标准 (Weight = 单步理想奖励)
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
            target_per_step = reward_term_cfg.weight

            # G. 判定：得分率 > 80%
            # 例如：5.5 (表现) > 0.8 * 6.0 (4.8)
            if mean_performance > 0.8 * target_per_step:

                # --- 3. 难度提升 ---
                delta_command = torch.tensor([-0.1, 0.1], device=env.device)
                new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
                new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

                new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
                new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

                base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
                base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

                print(f"[Curriculum] Lin Vel Level Up! Range: {base_velocity_ranges.lin_vel_x}, "
                      f"Perf: {mean_performance:.3f}/{target_per_step:.3f}")

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """
    修正版：基于 '平均每步得分 (Average Step Reward)' 的课程升级。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化 (保持不变) ---
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # --- 2. 检查逻辑 ---
    if env.common_step_counter % env.max_episode_length == 0 and env.common_step_counter > 0:

        # A. 分子：累积奖励
        current_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]

        # B. 分母：当前存活步数 (❌ 不要乘 dt)
        current_steps = env.episode_length_buf[env_ids]

        # C. 过滤噪音
        valid_mask = current_steps > 10

        if valid_mask.sum() > 0:
            # D. 计算：平均每步得分
            # 修正：直接除以步数，单位回归到 [Reward Value]
            avg_reward_per_step = current_sums[valid_mask] / current_steps[valid_mask]

            # E. 平均表现
            mean_performance = torch.mean(avg_reward_per_step)

            # F. 满分标准 (Weight = 单步理想奖励)
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
            target_per_step = reward_term_cfg.weight

            # G. 判定
            if mean_performance > 0.8 * target_per_step:

                # --- 3. 难度提升 ---
                delta_command = torch.tensor([-0.1, 0.1], device=env.device)
                new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

                new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

                base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

                print(f"[Curriculum] Ang Vel Level Up! Range: {base_velocity_ranges.ang_vel_z}, "
                      f"Perf: {mean_performance:.3f}/{target_per_step:.3f}")

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)