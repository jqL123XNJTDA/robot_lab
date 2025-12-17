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


def _init_lin_vel_curriculum(
    env: ManagerBasedRLEnv,
    base_velocity_ranges,
    range_multiplier: Sequence[float],
) -> None:
    """初始化线速度课程学习状态（支持 Resume）。

    使用 hasattr 检查，确保 Resume 时也能正确初始化。
    Resume 时从当前 base_velocity_ranges 的值继续，而非强制重置。
    """
    # 检查是否已初始化（避免重复初始化）
    if hasattr(env, "_curriculum_lin_vel_initialized") and env._curriculum_lin_vel_initialized:
        return

    # 保存原始配置范围（从配置文件读取的值）
    config_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
    config_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

    # 计算初始和最终范围边界
    # 注意：这里假设配置文件中的值是"目标最终范围"
    env._original_vel_x = config_vel_x.clone()
    env._original_vel_y = config_vel_y.clone()
    env._initial_vel_x = config_vel_x * range_multiplier[0]
    env._final_vel_x = config_vel_x * range_multiplier[1]
    env._initial_vel_y = config_vel_y * range_multiplier[0]
    env._final_vel_y = config_vel_y * range_multiplier[1]

    # 判断是否是 Resume（通过检查当前范围是否在合理区间内）
    is_resume = env.common_step_counter > 0

    # Resume 或新训练：都从初始范围开始（课程进度不持久化）
    if is_resume:
        print(f"[Curriculum] Resume detected (step={env.common_step_counter}). "
              f"Resetting lin_vel curriculum to initial range.")

    base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
    base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # 初始化性能历史记录（滑动窗口）
    env._lin_vel_perf_history = []

    # 标记已初始化
    env._curriculum_lin_vel_initialized = True

    print(f"[Curriculum] Lin Vel initialized: "
          f"initial={env._initial_vel_x.tolist()}, final={env._final_vel_x.tolist()}")


def _init_ang_vel_curriculum(
    env: ManagerBasedRLEnv,
    base_velocity_ranges,
    range_multiplier: Sequence[float],
) -> None:
    """初始化角速度课程学习状态（支持 Resume）。"""
    if hasattr(env, "_curriculum_ang_vel_initialized") and env._curriculum_ang_vel_initialized:
        return

    config_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)

    env._original_ang_vel_z = config_ang_vel_z.clone()
    env._initial_ang_vel_z = config_ang_vel_z * range_multiplier[0]
    env._final_ang_vel_z = config_ang_vel_z * range_multiplier[1]

    is_resume = env.common_step_counter > 0

    if is_resume:
        print(f"[Curriculum] Resume detected (step={env.common_step_counter}). "
              f"Resetting ang_vel curriculum to initial range.")
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()
    else:
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    env._ang_vel_perf_history = []
    env._curriculum_ang_vel_initialized = True

    print(f"[Curriculum] Ang Vel initialized: "
          f"initial={env._initial_ang_vel_z.tolist()}, final={env._final_ang_vel_z.tolist()}")


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    performance_threshold: float = 0.8,
    delta_ratio: float = 0.1,
    history_window: int = 10,
) -> torch.Tensor:
    """
    基于滑动窗口平均性能的线速度课程学习。

    支持 Resume：
    - 使用 hasattr 检查初始化状态，而非 common_step_counter == 0
    - Resume 时从初始范围重新开始（课程进度不持久化）
    - 如需持久化进度，需在 checkpoint 保存/加载时处理

    Args:
        env: 环境实例
        env_ids: 刚完成 episode 的环境 ID（由 CurriculumManager 提供）
        reward_term_name: 用于评估性能的奖励项名称
        range_multiplier: 初始/最终范围乘数 [初始比例, 最终比例]
        performance_threshold: 升级阈值（相对于满分的比例）
        delta_ratio: 每次升级的范围扩展比例
        history_window: 性能历史窗口大小

    Returns:
        当前线速度范围上界（用于日志记录）
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化（支持 Resume）---
    _init_lin_vel_curriculum(env, base_velocity_ranges, range_multiplier)

    # --- 2. 检测 episode 完成 ---
    if len(env_ids) > 0:
        # A. 获取累积奖励（在重置前获取）
        current_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]

        # B. 获取存活步数
        current_steps = env.episode_length_buf[env_ids].float()

        # C. 过滤短寿命环境
        valid_mask = current_steps > 10

        if valid_mask.sum() > 0:
            # D. 计算平均每步得分
            avg_reward_per_step = current_sums[valid_mask] / current_steps[valid_mask]

            # E. 计算这批环境的平均表现
            batch_mean_performance = torch.mean(avg_reward_per_step).item()

            # F. 添加到滑动窗口
            env._lin_vel_perf_history.append(batch_mean_performance)
            if len(env._lin_vel_perf_history) > history_window:
                env._lin_vel_perf_history.pop(0)

            # G. 窗口填满后评估升级
            if len(env._lin_vel_perf_history) >= history_window:
                window_mean_performance = sum(env._lin_vel_perf_history) / len(env._lin_vel_perf_history)

                reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
                target_per_step = reward_term_cfg.weight

                # H. 判定升级
                if window_mean_performance > performance_threshold * target_per_step:
                    current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
                    current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

                    # 计算相对 delta
                    span_x = env._final_vel_x[1] - env._final_vel_x[0]
                    span_y = env._final_vel_y[1] - env._final_vel_y[0]
                    delta_x = span_x * delta_ratio
                    delta_y = span_y * delta_ratio

                    # 扩展范围
                    new_vel_x = torch.tensor([
                        current_vel_x[0] - delta_x * 0.5,
                        current_vel_x[1] + delta_x * 0.5,
                    ], device=env.device)
                    new_vel_y = torch.tensor([
                        current_vel_y[0] - delta_y * 0.5,
                        current_vel_y[1] + delta_y * 0.5,
                    ], device=env.device)

                    # 分别 clamp 上下界
                    new_vel_x[0] = torch.clamp(new_vel_x[0], min=env._final_vel_x[0], max=env._initial_vel_x[0])
                    new_vel_x[1] = torch.clamp(new_vel_x[1], min=env._initial_vel_x[1], max=env._final_vel_x[1])
                    new_vel_y[0] = torch.clamp(new_vel_y[0], min=env._final_vel_y[0], max=env._initial_vel_y[0])
                    new_vel_y[1] = torch.clamp(new_vel_y[1], min=env._initial_vel_y[1], max=env._final_vel_y[1])

                    # 检查是否有变化
                    if not torch.allclose(new_vel_x, current_vel_x) or not torch.allclose(new_vel_y, current_vel_y):
                        base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
                        base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

                        env._lin_vel_perf_history.clear()

                        print(f"[Curriculum] Lin Vel Level Up! "
                              f"X: {base_velocity_ranges.lin_vel_x}, Y: {base_velocity_ranges.lin_vel_y}, "
                              f"Perf: {window_mean_performance:.3f}/{target_per_step:.3f}")

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    performance_threshold: float = 0.8,
    delta_ratio: float = 0.1,
    history_window: int = 10,
) -> torch.Tensor:
    """
    基于滑动窗口平均性能的角速度课程学习。

    支持 Resume：同 command_levels_lin_vel。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化（支持 Resume）---
    _init_ang_vel_curriculum(env, base_velocity_ranges, range_multiplier)

    # --- 2. 检测 episode 完成 ---
    if len(env_ids) > 0:
        current_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
        current_steps = env.episode_length_buf[env_ids].float()
        valid_mask = current_steps > 10

        if valid_mask.sum() > 0:
            avg_reward_per_step = current_sums[valid_mask] / current_steps[valid_mask]
            batch_mean_performance = torch.mean(avg_reward_per_step).item()

            env._ang_vel_perf_history.append(batch_mean_performance)
            if len(env._ang_vel_perf_history) > history_window:
                env._ang_vel_perf_history.pop(0)

            if len(env._ang_vel_perf_history) >= history_window:
                window_mean_performance = sum(env._ang_vel_perf_history) / len(env._ang_vel_perf_history)

                reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
                target_per_step = reward_term_cfg.weight

                if window_mean_performance > performance_threshold * target_per_step:
                    current_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)

                    span_z = env._final_ang_vel_z[1] - env._final_ang_vel_z[0]
                    delta_z = span_z * delta_ratio

                    new_ang_vel_z = torch.tensor([
                        current_ang_vel_z[0] - delta_z * 0.5,
                        current_ang_vel_z[1] + delta_z * 0.5,
                    ], device=env.device)

                    new_ang_vel_z[0] = torch.clamp(new_ang_vel_z[0], min=env._final_ang_vel_z[0], max=env._initial_ang_vel_z[0])
                    new_ang_vel_z[1] = torch.clamp(new_ang_vel_z[1], min=env._initial_ang_vel_z[1], max=env._final_ang_vel_z[1])

                    if not torch.allclose(new_ang_vel_z, current_ang_vel_z):
                        base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()
                        env._ang_vel_perf_history.clear()

                        print(f"[Curriculum] Ang Vel Level Up! Range: {base_velocity_ranges.ang_vel_z}, "
                              f"Perf: {window_mean_performance:.3f}/{target_per_step:.3f}")

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
