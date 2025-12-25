"""Script to play a trained policy with RayCasterCamera depth visualization."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 添加 rsl_rl 脚本路径
RL_SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts/reinforcement_learning/rsl_rl"))
RL_UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts/reinforcement_learning"))
sys.path.insert(0, RL_SCRIPTS_PATH)
sys.path.insert(0, RL_UTILS_PATH)
import cli_args  # isort: skip
from rl_utils import camera_follow  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play trained policy with depth camera visualization.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard for velocity commands.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import gymnasium as gym
import time
import torch
import numpy as np
import torchvision

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
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

# ========== 深度图处理参数 ==========
CLIP_RANGE = (0.3, 3.0)
RESIZE = (87, 58)
resize_transform = torchvision.transforms.Resize(
    (RESIZE[1], RESIZE[0]),
    interpolation=torchvision.transforms.InterpolationMode.BICUBIC
).to('cuda')

def _process_depth_image(depth_image):
    """处理深度图像，用于神经网络输入"""
    depth_image = torch.from_numpy(depth_image[:, :, 0]).to('cuda')
    depth_image = _crop_depth_image(depth_image)
    depth_image = resize_transform(depth_image[None, :]).squeeze()
    depth_image = _normalize_depth_image(depth_image)
    return depth_image.detach().cpu().numpy()

def _crop_depth_image(depth_image):
    return depth_image[:-2, 4:-4]

def _normalize_depth_image(depth_image):
    # 先 clip 到有效范围，防止归一化结果超出 [-0.5, 0.5]
    depth_image = torch.clamp(depth_image, CLIP_RANGE[0], CLIP_RANGE[1])
    depth_image = (depth_image - CLIP_RANGE[0]) / (CLIP_RANGE[1] - CLIP_RANGE[0]) - 0.5
    return depth_image


def visualize_depth(env, count: int, output_dir: str):
    """可视化深度相机数据"""
    # 检查场景中是否有相机传感器
    if "depth_camera" not in env.unwrapped.scene.sensors:
        return

    camera = env.unwrapped.scene.sensors["depth_camera"]
    depth_data = camera.data.output["distance_to_image_plane"]

    # 获取图像尺寸
    img_height = camera.cfg.pattern_cfg.height
    img_width = camera.cfg.pattern_cfg.width
    num_envs = depth_data.shape[0]

    # reshape 为图像格式并转置（修复竖直方向问题）
    if len(depth_data.shape) == 2:
        depth_camera = depth_data.reshape(num_envs, img_width, img_height).detach().cpu().numpy()
        depth_camera = np.transpose(depth_camera, (0, 2, 1))
    else:
        depth_camera = depth_data.squeeze(-1).detach().cpu().numpy()

    # 只显示第一个环境的深度图
    depth_image = depth_camera[0]

    # 调试：打印深度范围
    if count % 50 == 0:
        valid_depth = depth_image[~np.isnan(depth_image) & ~np.isinf(depth_image)]
        if len(valid_depth) > 0:
            print(f"[DEBUG] Depth range: min={valid_depth.min():.3f}, max={valid_depth.max():.3f}, mean={valid_depth.mean():.3f}")
        else:
            print("[DEBUG] No valid depth values!")

    # 处理 nan/inf 值
    depth_image = np.nan_to_num(depth_image, nan=1.5, posinf=1.5, neginf=0.0)

    # 尝试显示图像
    try:
        # 原始深度图显示（0-1.5m 映射到 colormap）
        depth_colormap = cv2.applyColorMap(
            (depth_image / 3 * 255).clip(0, 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imshow('depth_camera', depth_colormap)

        # 处理后的深度图（为神经网络）
        process_image = _process_depth_image(depth_image[:, :, None])
        cv2.imshow('process_image', process_image + 0.5)  # 偏移0.5使可视化更好
        cv2.waitKey(1)
    except cv2.error:
        # OpenCV 没有 GUI 支持，保存图像到文件
        if count % 100 == 0:
            depth_colormap = cv2.applyColorMap(
                (depth_image / 1.5 * 255).clip(0, 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(os.path.join(output_dir, f'depth_{count}.png'), depth_colormap)
            process_image = _process_depth_image(depth_image[:, :, None])
            cv2.imwrite(os.path.join(output_dir, f'processed_{count}.png'),
                       ((process_image + 0.5) * 255).astype(np.uint8))
            print(f"[INFO]: Saved depth images at frame {count}")


def add_depth_camera_to_env_cfg(env_cfg):
    """在环境配置中添加深度相机（在环境创建前调用）"""
    # 获取地形的 prim_path
    terrain_prim_path = "/World/ground"
    if hasattr(env_cfg.scene, 'terrain') and hasattr(env_cfg.scene.terrain, 'prim_path'):
        terrain_prim_path = env_cfg.scene.terrain.prim_path

    # 创建相机配置
    camera_cfg = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.05,  # 20Hz
        data_types=["distance_to_image_plane"],
        debug_vis=True,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.6, 0.0, 0.1),
            rot = (0.9659, 0.0, 0.2588, 0.0),  # 朝下 30°
            convention = "world",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=47.5,  # 水平 FOV ≈ 87°
            width=106,
            height=60,
        ),
        max_distance=3,
        mesh_prim_paths=[terrain_prim_path],
    )

    # 添加到场景配置
    env_cfg.scene.depth_camera = camera_cfg
    print(f"[INFO] Depth camera configured with mesh_prim_paths: {camera_cfg.mesh_prim_paths}")
    return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and visualize depth camera."""
    # 获取任务名称
    task_name = args_cli.task.split(":")[-1]

    # 更新配置
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 4

    # 设置环境种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 设置地形参数（减少内存使用）
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # 禁用随机化（用于评估）
    env_cfg.observations.policy.enable_corruption = False
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

    # 查找检查点路径
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # 在创建环境前添加深度相机配置
    has_depth_camera = add_depth_camera_to_env_cfg(env_cfg)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 创建环境
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 转换为单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 加载策略
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # 获取推理策略
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 获取时间步长
    dt = env.unwrapped.step_dt

    # 重置环境
    obs = env.get_observations()
    timestep = 0

    print("[INFO] ========================================")
    print("[INFO] 策略推理 + 深度相机可视化已启动")
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

            # 深度相机可视化
            if has_depth_camera:
                visualize_depth(env, timestep, output_dir)

        timestep += 1

        # 键盘模式下相机跟随
        if args_cli.keyboard:
            camera_follow(env)

        # 实时运行延时
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # OpenCV 按键检测（用于退出）
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键退出
            print("[INFO] ESC pressed, exiting...")
            break

    # 关闭环境
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
