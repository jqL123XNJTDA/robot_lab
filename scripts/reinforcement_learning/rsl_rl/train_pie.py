# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) 训练脚本
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 用法:
#   python scripts/reinforcement_learning/rsl_rl/train_pie.py \
#       --task=RobotLab-Isaac-Velocity-PIE-Flat-MyDog-v0 \
#       --headless \
#       --num_envs=4096

"""Script to train RL agent with PIE framework."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with PIE framework.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="pie_cfg_entry_point", help="Name of the PIE agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

# 导入PIE Runner
from runners.pie_on_policy_runner import PIEOnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def get_pie_cfg(task_name: str, agent_entry_point: str):
    """获取PIE配置

    Args:
        task_name: 任务名称
        agent_entry_point: Agent配置入口点名称

    Returns:
        env_cfg: 环境配置
        agent_cfg: PIE Runner配置
    """
    # 获取环境规格
    env_spec = gym.spec(task_name)
    if env_spec is None:
        raise ValueError(f"Task '{task_name}' not found in gym registry.")

    # 获取环境配置
    env_cfg_entry_point = env_spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_entry_point is None:
        raise ValueError(f"Task '{task_name}' does not have 'env_cfg_entry_point' in kwargs.")

    # 动态导入环境配置
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
    """Train with PIE framework."""
    # 获取配置
    env_cfg, agent_cfg = get_pie_cfg(args_cli.task, args_cli.agent)

    # 覆盖命令行参数
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    # 设置设备
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 检查CPU + 分布式训练的无效组合
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # 多GPU训练配置
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 不同进程使用不同种子
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 运行目录: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if hasattr(agent_cfg, 'run_name') and agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 设置环境日志目录
    env_cfg.log_dir = log_dir

    # 创建Isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 转换为单Agent环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 检查是否需要恢复训练
    resume_path = None
    if args_cli.resume:
        load_run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        load_checkpoint = args_cli.checkpoint if args_cli.checkpoint else agent_cfg.load_checkpoint
        resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)
        print(f"[INFO] Resuming from checkpoint: {resume_path}")

    # 视频录制包装
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL环境包装
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建PIE Runner
    runner = PIEOnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # 记录git状态
    runner.add_git_repo_to_log(__file__)

    # 加载检查点
    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # 保存配置
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 开始训练
    print("[INFO] Starting PIE training...")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
