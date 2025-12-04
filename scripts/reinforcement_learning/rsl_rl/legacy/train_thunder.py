#!/usr/bin/env python3
"""
Train RL agent with Thunder Runner and Velocity Estimator.

This script uses:
1. ThunderRunner - Custom PPO runner with asymmetric actor-critic
2. VelocityEstimatorBarlowTwins - Network with self-supervised learning
3. ImplicitEncoder - Privileged information encoder

Features:
- Asymmetric Actor-Critic (Actor uses estimated velocity, Critic uses privileged info)
- Barlow Twins self-supervised learning
- Automatic mixed precision (AMP) training
- Episode tracking and detailed logging

Usage:
    python train_thunder.py \
        --task RobotLab-Isaac-Velocity-Rough-Thunder-v0 \
        --num_envs 4096 \
        --encoder_type gru \
        --use_barlow_twins \
        --headless
"""

import argparse
import os
import sys
from datetime import datetime

# Isaac Sim must be launched first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train RL agent with Thunder Runner.")

# Standard training arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")

# Velocity Estimator arguments
parser.add_argument(
    "--encoder_type",
    type=str,
    default="gru",
    choices=["gru", "transformer", "conv1d"],
    help="Type of encoder: 'gru' (fast), 'transformer', or 'conv1d'",
)
parser.add_argument(
    "--history_len",
    type=int,
    default=10,
    help="Number of frames in observation history (10 frames = 0.4s at 25Hz)",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=12,
    help="Latent dimension (should match privileged features, default 12)",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=128,
    help="Hidden dimension for encoder",
)
parser.add_argument(
    "--estimator_lr",
    type=float,
    default=1e-3,
    help="Learning rate for velocity estimator",
)
parser.add_argument(
    "--use_barlow_twins",
    action="store_true",
    default=True,
    help="Use Barlow Twins self-supervised learning",
)
parser.add_argument(
    "--no_barlow_twins",
    action="store_true",
    default=False,
    help="Disable Barlow Twins",
)
parser.add_argument(
    "--barlow_weight",
    type=float,
    default=0.1,
    help="Weight for Barlow Twins loss",
)
parser.add_argument(
    "--disable_estimator",
    action="store_true",
    default=False,
    help="Disable velocity estimator (for baseline comparison)",
)

# Import cli_args after parser is defined
import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Handle barlow twins flag
if args_cli.no_barlow_twins:
    args_cli.use_barlow_twins = False

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'."
    )
    exit(1)

"""Rest everything follows."""
import gymnasium as gym
import torch

import omni
from rsl_rl.runners import DistillationRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

# Import Thunder Runner and Velocity Estimator Wrapper
from thunder_runner import create_thunder_runner
from velocity_estimator_wrapper import (
    create_velocity_estimator_wrapper,
    VelocityEstimatorConfig,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with Thunder Runner and Velocity Estimator."""
    
    print("\n" + "=" * 80)
    print("🎯 Thunder Runner Training with Velocity Estimator")
    print("=" * 80)
    
    # Override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Set environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Multi-GPU configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # Setup logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"📁 Logging to: {log_dir}")

    # Set IO descriptors export flag
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir

    # ========================================================================
    # Create environment
    # ========================================================================
    print("\n🌍 Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ========================================================================
    # 🔧 Create Velocity Estimator Wrapper
    # ========================================================================
    velocity_wrapper = None
    if not args_cli.disable_estimator:
        # Get observation dimension from environment's unwrapped observation manager
        obs = env.unwrapped.observation_manager.compute()
        if isinstance(obs, dict):
            policy_obs = obs['policy']
        else:
            policy_obs = obs
        obs_dim = policy_obs.shape[1]
        
        # Create velocity estimator config
        estimator_config = VelocityEstimatorConfig(
            obs_dim=obs_dim,
            history_len=args_cli.history_len,
            hidden_dim=args_cli.hidden_dim,
            latent_dim=args_cli.latent_dim,
            encoder_type=args_cli.encoder_type,
            use_barlow_twins=args_cli.use_barlow_twins,
            barlow_proj_dim=64,
            learning_rate=args_cli.estimator_lr,
            velocity_weight=1.0,
            latent_weight=1.0,
            barlow_weight=args_cli.barlow_weight,
            lambda_off_diag=5e-3,
            device=agent_cfg.device,
        )
        
        # Create wrapper
        velocity_wrapper = create_velocity_estimator_wrapper(
            env=env.unwrapped,
            obs_dim=obs_dim,
            history_len=args_cli.history_len,
            latent_dim=args_cli.latent_dim,
            encoder_type=args_cli.encoder_type,
            use_barlow_twins=args_cli.use_barlow_twins,
            learning_rate=args_cli.estimator_lr,
            device=agent_cfg.device,
        )
        print("✅ Velocity Estimator Wrapper enabled")
    else:
        print("⚠️  Velocity Estimator disabled (baseline mode)")

    # ========================================================================
    # Continue with standard training setup
    # ========================================================================

    # Save resume path
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_run and "/" in agent_cfg.load_run:
            load_path_parts = agent_cfg.load_run.split("/", 1)
            experiment_name_from_load = load_path_parts[0]
            run_dir_from_load = load_path_parts[1]
            parent_log_path = os.path.join("logs", "rsl_rl")
            parent_log_path = os.path.abspath(parent_log_path)
            resume_path = get_checkpoint_path(
                parent_log_path, experiment_name_from_load, agent_cfg.load_checkpoint, other_dirs=[run_dir_from_load]
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("\n📹 Video recording enabled")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ========================================================================
    # 🔧 Create Thunder Runner (with or without velocity estimator)
    # ========================================================================
    print("\n🏃 Creating Thunder Runner...")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = create_thunder_runner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            velocity_wrapper=velocity_wrapper,
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Write git state
    runner.add_git_repo_to_log(__file__)

    # Load checkpoint if resuming
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"📥 Loading checkpoint from: {resume_path}")
        runner.load(resume_path)
        
        # Also load estimator checkpoint if available
        if velocity_wrapper is not None:
            estimator_path = resume_path.replace("model_", "estimator_")
            if os.path.exists(estimator_path):
                velocity_wrapper.load(estimator_path)

    # Dump configurations
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Save estimator config
    if velocity_wrapper is not None:
        estimator_config_dict = {
            "obs_dim": obs_dim,
            "history_len": args_cli.history_len,
            "hidden_dim": args_cli.hidden_dim,
            "latent_dim": args_cli.latent_dim,
            "encoder_type": args_cli.encoder_type,
            "use_barlow_twins": args_cli.use_barlow_twins,
            "barlow_weight": args_cli.barlow_weight,
            "learning_rate": args_cli.estimator_lr,
        }
        dump_yaml(os.path.join(log_dir, "params", "estimator.yaml"), estimator_config_dict)

    print("\n" + "=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)
    print(f"  Task:           {args_cli.task}")
    print(f"  Num Envs:       {env_cfg.scene.num_envs}")
    print(f"  Max Iterations: {agent_cfg.max_iterations}")
    print(f"  Device:         {agent_cfg.device}")
    if velocity_wrapper is not None:
        print(f"  Estimator:      {args_cli.encoder_type.upper()}")
        print(f"  Barlow Twins:   {args_cli.use_barlow_twins}")
        print(f"  Latent Dim:     {args_cli.latent_dim}")
    print("=" * 80)
    print("\n📊 Monitor training with: tensorboard --logdir logs/rsl_rl/\n")

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Save final estimator checkpoint
    if velocity_wrapper is not None:
        estimator_save_path = os.path.join(log_dir, "estimator_final.pt")
        velocity_wrapper.save(estimator_save_path)

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run main function
    main()
    # Close sim app
    simulation_app.close()

