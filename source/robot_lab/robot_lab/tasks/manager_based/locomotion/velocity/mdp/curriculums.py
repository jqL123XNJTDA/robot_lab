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
    performance_threshold: float = 0.8,  # 相对于 weight 的比例
    delta_ratio: float = 0.1,
    history_window: int = 10,  # 滑动窗口大小
) -> torch.Tensor:
    """
    线速度课程学习 - 使用刚完成 episode 的环境来评估性能。

    Args:
        env: 环境实例
        env_ids: 刚完成 episode 的环境 ID（由 CurriculumManager 提供）
        reward_term_name: 奖励项名称 (如 "track_lin_vel_xy_exp")
        range_multiplier: [初始比例, 最终比例]
        performance_threshold: 升级阈值（相对于 weight）
        delta_ratio: 每次升级扩展的比例
        history_window: 滑动窗口大小
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化（支持 Resume）---
    if not hasattr(env, "_curriculum_lin_vel_initialized"):
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # 设置初始范围
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

        env._lin_vel_perf_history = []
        env._curriculum_lin_vel_initialized = True

        print(f"[Curriculum] Lin Vel initialized: "
              f"initial={env._initial_vel_x.tolist()}, final={env._final_vel_x.tolist()}")

    # --- 2. 只处理刚完成的 episode ---
    if len(env_ids) > 0:
        # 从 env.extras["log"] 获取 Episode_Reward（reward_manager.reset() 已计算好）
        log_key = f"Episode_Reward/{reward_term_name}"
        if log_key in env.extras.get("log", {}):
            episode_reward = env.extras["log"][log_key]
            if isinstance(episode_reward, torch.Tensor):
                episode_reward = episode_reward.item()

            # 过滤掉早期死亡的环境（episode_reward ≈ 0 表示没跑多久就死了）
            if episode_reward > 0.01:  # 只记录有效的 episode
                env._lin_vel_perf_history.append(episode_reward)
                if len(env._lin_vel_perf_history) > history_window:
                    env._lin_vel_perf_history.pop(0)

    # --- 3. 窗口填满后评估 ---
    if hasattr(env, "_lin_vel_perf_history") and len(env._lin_vel_perf_history) >= history_window:
        window_avg_reward = sum(env._lin_vel_perf_history) / len(env._lin_vel_perf_history)

        # 阈值：weight × threshold
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        threshold = reward_term_cfg.weight * performance_threshold

        # 每次评估打印（使用计数器控制频率）
        if not hasattr(env, "_lin_vel_eval_count"):
            env._lin_vel_eval_count = 0
        env._lin_vel_eval_count += 1
        if env._lin_vel_eval_count % 50 == 0:
            print(f"[Curriculum EVAL] Lin Vel: avg_reward={window_avg_reward:.4f}, "
                  f"threshold={threshold:.4f}, range_x={base_velocity_ranges.lin_vel_x}", flush=True)

        # --- 4. 判断升级 ---
        if window_avg_reward > threshold:
            current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
            current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

            # 计算扩展量
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

            # Clamp 到最终范围
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # 检查是否有变化
            if not torch.allclose(new_vel_x, current_vel_x):
                base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
                base_velocity_ranges.lin_vel_y = new_vel_y.tolist()
                env._lin_vel_perf_history.clear()  # 清空历史，重新积累

                print(f"[Curriculum] Lin Vel Level Up! "
                      f"X: {base_velocity_ranges.lin_vel_x}, "
                      f"Perf: {window_avg_reward:.4f}/{threshold:.4f}")

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
    performance_threshold: float = 0.8,  # 相对于 weight 的比例
    delta_ratio: float = 0.1,
    history_window: int = 10,  # 滑动窗口大小
) -> torch.Tensor:
    """
    角速度课程学习 - 使用刚完成 episode 的环境来评估性能。
    """
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # --- 1. 初始化（支持 Resume）---
    if not hasattr(env, "_curriculum_ang_vel_initialized"):
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # 设置初始范围
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

        env._ang_vel_perf_history = []
        env._curriculum_ang_vel_initialized = True

        print(f"[Curriculum] Ang Vel initialized: "
              f"initial={env._initial_ang_vel_z.tolist()}, final={env._final_ang_vel_z.tolist()}")

    # --- 2. 只处理刚完成的 episode ---
    if len(env_ids) > 0:
        log_key = f"Episode_Reward/{reward_term_name}"
        if log_key in env.extras.get("log", {}):
            episode_reward = env.extras["log"][log_key]
            if isinstance(episode_reward, torch.Tensor):
                episode_reward = episode_reward.item()

            # 过滤掉早期死亡的环境
            if episode_reward > 0.01:
                env._ang_vel_perf_history.append(episode_reward)
                if len(env._ang_vel_perf_history) > history_window:
                    env._ang_vel_perf_history.pop(0)

    # --- 3. 窗口填满后评估 ---
    if hasattr(env, "_ang_vel_perf_history") and len(env._ang_vel_perf_history) >= history_window:
        window_avg_reward = sum(env._ang_vel_perf_history) / len(env._ang_vel_perf_history)

        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        threshold = reward_term_cfg.weight * performance_threshold

        if not hasattr(env, "_ang_vel_eval_count"):
            env._ang_vel_eval_count = 0
        env._ang_vel_eval_count += 1
        if env._ang_vel_eval_count % 50 == 0:
            print(f"[Curriculum EVAL] Ang Vel: avg_reward={window_avg_reward:.4f}, "
                  f"threshold={threshold:.4f}, range_z={base_velocity_ranges.ang_vel_z}", flush=True)

        # --- 4. 判断升级 ---
        if window_avg_reward > threshold:
            current_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)

            span_z = env._final_ang_vel_z[1] - env._final_ang_vel_z[0]
            delta_z = span_z * delta_ratio

            new_ang_vel_z = torch.tensor([
                current_ang_vel_z[0] - delta_z * 0.5,
                current_ang_vel_z[1] + delta_z * 0.5,
            ], device=env.device)

            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            if not torch.allclose(new_ang_vel_z, current_ang_vel_z):
                base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()
                env._ang_vel_perf_history.clear()

                print(f"[Curriculum] Ang Vel Level Up! "
                      f"Z: {base_velocity_ranges.ang_vel_z}, "
                      f"Perf: {window_avg_reward:.4f}/{threshold:.4f}")

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
