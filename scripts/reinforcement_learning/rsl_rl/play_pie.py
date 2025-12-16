# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) 评估脚本
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 用法:
#   python scripts/reinforcement_learning/rsl_rl/play_pie.py \
#       --task=RobotLab-Isaac-Velocity-PIE-Flat-MyDog-v0 \
#       --num_envs=64

"""Script to evaluate trained PIE policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained PIE policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="pie_cfg_entry_point", help="Name of the PIE agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard for velocity commands.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments (resume, load_run, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import robot_lab.tasks  # noqa: F401

# 导入PIE Runner
from runners.pie_on_policy_runner import PIEOnPolicyRunner

# 导入相机跟随工具
from rl_utils import camera_follow

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_pie_cfg(task_name: str, agent_entry_point: str):
    """获取PIE配置"""
    env_spec = gym.spec(task_name)
    if env_spec is None:
        raise ValueError(f"Task '{task_name}' not found in gym registry.")

    # 获取环境配置
    env_cfg_entry_point = env_spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_entry_point is None:
        raise ValueError(f"Task '{task_name}' does not have 'env_cfg_entry_point' in kwargs.")

    import importlib
    module_name, class_name = env_cfg_entry_point.rsplit(":", 1)
    module = importlib.import_module(module_name)
    env_cfg = getattr(module, class_name)()

    # 获取PIE Agent配置
    pie_cfg_entry_point = env_spec.kwargs.get(agent_entry_point)
    if pie_cfg_entry_point is None:
        raise ValueError(f"Task '{task_name}' does not have '{agent_entry_point}' in kwargs.")

    module_name, class_name = pie_cfg_entry_point.rsplit(":", 1)
    module = importlib.import_module(module_name)
    agent_cfg = getattr(module, class_name)()

    return env_cfg, agent_cfg


def main():
    """Evaluate trained PIE policy."""
    # 获取配置
    env_cfg, agent_cfg = get_pie_cfg(args_cli.task, args_cli.agent)

    # 设置环境数量
    env_cfg.scene.num_envs = args_cli.num_envs

    # 设置种子
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    # 设置设备
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 评估模式配置
    env_cfg.observations.policy.enable_corruption = False

    # 减少地形复杂度 (可选)
    if hasattr(env_cfg.scene, 'terrain') and hasattr(env_cfg.scene.terrain, 'terrain_generator'):
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    # 禁用随机化
    if hasattr(env_cfg, 'events'):
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.push_robot = None
    if hasattr(env_cfg, 'curriculum'):
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None

    # 键盘控制配置
    controller = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        if hasattr(env_cfg.commands, 'base_velocity'):
            env_cfg.commands.base_velocity.debug_vis = False
            config = Se2KeyboardCfg(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
            )
            controller = Se2Keyboard(config)
            env_cfg.observations.policy.velocity_commands = ObsTerm(
                func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
            )

    # 日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # 获取检查点路径
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        load_run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        load_checkpoint = "model_.*.pt"  # 默认加载最新模型
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # 创建环境
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 转换为单Agent环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 视频录制包装
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL环境包装
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建PIE Runner (评估模式)
    runner = PIEOnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=None,  # 评估模式不记录日志
        device=agent_cfg.device,
    )

    # 加载检查点
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # 获取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 获取时间步长
    dt = env.unwrapped.step_dt

    # 重置环境
    obs = env.get_observations()
    timestep = 0

    print("[INFO] ========================================")
    print("[INFO] PIE策略评估已启动")
    if args_cli.keyboard:
        print("[INFO] 键盘控制已启用:")
        print("  W/S - 前进/后退")
        print("  A/D - 左转/右转")
        print("  Q/E - 左移/右移")
    print("[INFO] ========================================")

    # 主循环
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # 策略推理
            actions = policy(obs)

            # 环境步进
            obs, _, _, _ = env.step(actions)

        timestep += 1

        # 键盘模式下相机跟随
        if args_cli.keyboard:
            camera_follow(env)

        # 实时运行延时
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
