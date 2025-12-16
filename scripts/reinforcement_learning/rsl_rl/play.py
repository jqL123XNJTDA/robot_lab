# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip
from rl_utils import camera_follow

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        # 键盘控制：替换 velocity_commands 观测，保持 clip/scale 参数与训练一致
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env, ctrl=controller: torch.tensor(ctrl.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        print("[INFO] 键盘控制已启用: W/S=前进/后退, A/D=左转/右转, Q/E=左移/右移")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # ========== 后腿关节限位监控 (已禁用) ==========
    hind_joint_monitor = None
    # ====================================

    # ========== 后腿Calf高度监控 (已禁用) ==========
    calf_height_monitor = None
    # ====================================

    # ========== 接触力监控 (已禁用) ==========
    contact_force_monitor = None
    # ====================================

    # ========== Base Link 高度监控 (已禁用) ==========
    base_height_monitor = None
    # ====================================

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # ========== 键盘调试 ==========
            if args_cli.keyboard and timestep % 50 == 0:
                # 直接调用 controller.advance() 检查键盘输入
                cmd = controller.advance()
                print(f"[DEBUG][Step {timestep}] Keyboard cmd: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")
            # ==============================

            # ========== 后腿关节限位检测 ==========
            if hind_joint_monitor is not None:
                monitor_asset = hind_joint_monitor["asset"]
                joint_ids = hind_joint_monitor["joint_ids"]  # 现在是 tensor
                tolerance = hind_joint_monitor["tolerance"]
                # 使用 tensor 索引，保持在同一设备
                joint_pos = monitor_asset.data.joint_pos[:, joint_ids]
                lower_limits = monitor_asset.data.soft_joint_pos_limits[:, joint_ids, 0]
                upper_limits = monitor_asset.data.soft_joint_pos_limits[:, joint_ids, 1]
                violation_mask = (joint_pos < (lower_limits - tolerance)) | (joint_pos > (upper_limits + tolerance))
                prev_violation = hind_joint_monitor["prev_violation"]
                new_violation = violation_mask & (~prev_violation)
                if new_violation.any():
                    # 只在需要打印时转到 CPU
                    joint_pos_cpu = joint_pos.detach().cpu()
                    lower_cpu = lower_limits.detach().cpu()
                    upper_cpu = upper_limits.detach().cpu()
                    new_violation_cpu = new_violation.detach().cpu()
                    for env_id, joint_idx in torch.nonzero(new_violation_cpu, as_tuple=False):
                        env_id = int(env_id.item())
                        joint_idx = int(joint_idx.item())
                        joint_name = hind_joint_monitor["joint_names"][joint_idx]
                        pos_val = joint_pos_cpu[env_id, joint_idx].item()
                        lower_val = lower_cpu[env_id, joint_idx].item()
                        upper_val = upper_cpu[env_id, joint_idx].item()
                        if pos_val < lower_val - tolerance:
                            limit_type = "下限"
                            limit_val = lower_val
                        elif pos_val > upper_val + tolerance:
                            limit_type = "上限"
                            limit_val = upper_val
                        else:
                            continue
                        print(
                            f"[WARN][Step {timestep}] Env {env_id} {joint_name} 超出{limit_type} "
                            f"(pos={pos_val:.3f} rad, limit={limit_val:.3f} rad)"
                        )
                hind_joint_monitor["prev_violation"] = violation_mask
            # ====================================

            # ========== 后腿Calf高度打印 ==========
            if calf_height_monitor is not None:
                calf_height_monitor["step_counter"] += 1
                if calf_height_monitor["step_counter"] % calf_height_monitor["print_interval"] == 0:
                    monitor_asset = calf_height_monitor["asset"]
                    body_ids = calf_height_monitor["body_ids"]
                    body_names = calf_height_monitor["body_names"]

                    # 获取小腿的世界坐标位置 (num_envs, num_calves, 3)
                    calf_pos_w = monitor_asset.data.body_pos_w[:, body_ids, :]
                    # 提取Z轴高度 (num_envs, num_calves)
                    calf_heights = calf_pos_w[:, :, 2].detach().cpu()

                    # 打印Env 0的高度信息
                    env0_heights = calf_heights[0]
                    print(f"[INFO][Step {timestep}] Env 0 后腿Calf高度:")
                    for i, name in enumerate(body_names):
                        print(f"  {name}: {env0_heights[i]:.3f} m")

                    # 打印所有环境的统计信息
                    mean_heights = calf_heights.mean(dim=0)  # 对所有环境求平均
                    max_heights = calf_heights.max(dim=0).values
                    min_heights = calf_heights.min(dim=0).values
                    print(f"[INFO][Step {timestep}] 所有环境后腿Calf高度统计:")
                    for i, name in enumerate(body_names):
                        print(f"  {name}: mean={mean_heights[i]:.3f} m, min={min_heights[i]:.3f} m, max={max_heights[i]:.3f} m")
            # ====================================

            # ========== 接触力监控打印 ==========
            if contact_force_monitor is not None:
                contact_force_monitor["step_counter"] += 1
                sensor = contact_force_monitor["sensor"]
                body_ids = contact_force_monitor["body_ids"]
                body_names = contact_force_monitor["body_names"]
                threshold = contact_force_monitor["threshold"]
                foot_threshold = contact_force_monitor["foot_threshold"]

                # 获取接触力 (num_envs, history_len, num_bodies, 3)
                # 只取最新一帧 [:, -1, ...]
                net_forces = sensor.data.net_forces_w_history[:, -1, body_ids, :]
                # 计算力的模长 (num_envs, num_bodies)
                force_magnitude = torch.norm(net_forces, dim=-1).detach().cpu()

                # 根据刚体类型使用不同阈值检测超限
                # foot 使用 foot_threshold (150N), 其他使用 threshold (10N)
                exceeds_threshold = torch.zeros_like(force_magnitude, dtype=torch.bool)
                for i, name in enumerate(body_names):
                    if "foot" in name.lower():
                        exceeds_threshold[:, i] = force_magnitude[:, i] > foot_threshold
                    else:
                        exceeds_threshold[:, i] = force_magnitude[:, i] > threshold

                # 实时打印超限警告 (只打印 Env 0)
                if exceeds_threshold[0].any():
                    for i, name in enumerate(body_names):
                        if exceeds_threshold[0, i]:
                            curr_threshold = foot_threshold if "foot" in name.lower() else threshold
                            print(f"[CONTACT][Step {timestep}] Env 0 {name} 接触力超限: {force_magnitude[0, i]:.2f} N > {curr_threshold} N")

                # 定期打印统计信息
                if contact_force_monitor["step_counter"] % contact_force_monitor["print_interval"] == 0:
                    # 统计所有环境中超限的数量
                    num_exceeds = exceeds_threshold.sum(dim=0)  # (num_bodies,)
                    total_envs = force_magnitude.shape[0]

                    print(f"[INFO][Step {timestep}] 接触力统计 (base/calf阈值={threshold}N, foot阈值={foot_threshold}N):")
                    for i, name in enumerate(body_names):
                        mean_force = force_magnitude[:, i].mean().item()
                        max_force = force_magnitude[:, i].max().item()
                        exceed_ratio = num_exceeds[i].item() / total_envs * 100
                        curr_threshold = foot_threshold if "foot" in name.lower() else threshold
                        print(f"  {name}: mean={mean_force:.2f}N, max={max_force:.2f}N, 超限率={exceed_ratio:.1f}% (>{curr_threshold}N)")
            # ====================================

            # ========== Base Link 高度打印 ==========
            if base_height_monitor is not None:
                base_height_monitor["step_counter"] += 1
                if base_height_monitor["step_counter"] % base_height_monitor["print_interval"] == 0:
                    monitor_asset = base_height_monitor["asset"]
                    body_id = base_height_monitor["body_id"]
                    body_name = base_height_monitor["body_name"]

                    # 获取 base_link 的世界坐标位置 (num_envs, 3)
                    base_pos_w = monitor_asset.data.body_pos_w[:, body_id, :]
                    # 提取 Z 轴高度 (num_envs,)
                    base_heights = base_pos_w[:, 2].detach().cpu()

                    # 打印 Env 0 的高度
                    env0_height = base_heights[0].item()
                    
                    # 打印所有环境的统计信息
                    mean_height = base_heights.mean().item()
                    max_height = base_heights.max().item()
                    min_height = base_heights.min().item()
                    
                    print(f"[INFO][Step {timestep}] {body_name} 高度: "
                          f"Env0={env0_height:.3f}m, mean={mean_height:.3f}m, "
                          f"min={min_height:.3f}m, max={max_height:.3f}m")
            # ====================================

        timestep += 1  # 始终递增 timestep 用于调试

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
