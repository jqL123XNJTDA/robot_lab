# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# PIE (Parkour with Implicit-Explicit Learning Framework) On-Policy Runner
# 参考论文: "PIE: Parkour With Implicit-Explicit Learning Framework for Legged Robots"
# IEEE Robotics and Automation Letters, 2024
#
# 负责训练循环、观测收集、历史管理

import os
import time
import statistics
import torch
from collections import deque
from typing import Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from modules.pie_actor_critic import PIEActorCritic, create_pie_actor_critic
from modules.pie_estimator import PIEEstimatorConfig
from algorithms.pie_ppo import PIEPPO
from storage.pie_rollout_storage import PIERolloutStorage, PIETransition


class PIEOnPolicyRunner:
    """PIE On-Policy Runner

    负责：
    1. 观测收集和历史管理（深度图、本体感受）
    2. 与环境交互
    3. 数据存储和PPO更新
    4. TensorBoard日志
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        """初始化Runner

        Args:
            env: Isaac Lab环境
            train_cfg: 训练配置字典
            log_dir: 日志目录
            device: 设备
        """
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # 提取配置
        self.num_envs = env.num_envs
        self.num_steps_per_env = train_cfg.get("num_steps_per_env", 24)
        self.max_iterations = train_cfg.get("max_iterations", 10000)
        self.save_interval = train_cfg.get("save_interval", 100)

        # PIE配置 - 从estimator字典获取
        estimator_cfg = train_cfg.get("estimator", {})
        self.num_depth_frames = estimator_cfg.get("num_depth_frames", 2)
        self.num_proprio_frames = estimator_cfg.get("num_proprio_frames", 10)
        self.proprio_dim = estimator_cfg.get("proprio_dim", 57)
        self.latent_dim = estimator_cfg.get("latent_dim", 32)
        self.height_map_dim = estimator_cfg.get("height_map_dim", 187)
        self.foot_clearance_dim = estimator_cfg.get("foot_clearance_dim", 4)
        self.depth_height = estimator_cfg.get("depth_height", 58)
        self.depth_width = estimator_cfg.get("depth_width", 87)

        # 动作维度
        self.num_actions = env.num_actions

        # 观测维度（从环境获取）
        self.critic_obs_dim = self._get_critic_obs_dim()

        # 日志
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # 创建Actor-Critic
        self.actor_critic = self._create_actor_critic(train_cfg)

        # 创建存储
        self.storage = self._create_storage()

        # 创建PPO算法
        self.alg = self._create_algorithm(train_cfg)

        # 历史缓冲区
        self._init_history_buffers()

        # 训练统计
        self.current_iteration = 0
        self.total_timesteps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # 详细日志的额外缓冲区
        self.ep_infos = []  # 存储episode info用于详细日志
        self.tot_time = 0.0
        self.current_learning_iteration = 0

    def _get_critic_obs_dim(self) -> int:
        """获取Critic观测维度"""
        # 尝试从环境获取
        if hasattr(self.env, "observation_manager"):
            critic_dim = 0
            for group_name in ["critic", "height_scan_group"]:
                if group_name in self.env.observation_manager.group_obs_dim:
                    critic_dim += self.env.observation_manager.group_obs_dim[group_name]
            if critic_dim > 0:
                return critic_dim

        # 默认值
        return self.proprio_dim + 3 + self.height_map_dim

    def _create_actor_critic(self, cfg: dict) -> PIEActorCritic:
        """创建Actor-Critic"""
        estimator_cfg_dict = cfg.get("estimator", {})
        policy_cfg = cfg.get("policy", {})

        estimator_cfg = PIEEstimatorConfig(
            num_depth_frames=self.num_depth_frames,
            num_proprio_frames=self.num_proprio_frames,
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            height_map_dim=self.height_map_dim,
            foot_clearance_dim=self.foot_clearance_dim,
            depth_height=self.depth_height,
            depth_width=self.depth_width,
            learning_rate=estimator_cfg_dict.get("learning_rate", 1e-3),
        )

        return PIEActorCritic(
            estimator_cfg=estimator_cfg,
            num_actions=self.num_actions,
            actor_hidden_dims=tuple(policy_cfg.get("actor_hidden_dims", [512, 256, 128])),
            critic_obs_dim=self.critic_obs_dim,
            critic_hidden_dims=tuple(policy_cfg.get("critic_hidden_dims", [512, 256, 128])),
            activation=policy_cfg.get("activation", "elu"),
            init_noise_std=policy_cfg.get("init_noise_std", 1.0),
        )

    def _create_storage(self) -> PIERolloutStorage:
        """创建存储"""
        return PIERolloutStorage(
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            num_depth_frames=self.num_depth_frames,
            depth_height=self.depth_height,
            depth_width=self.depth_width,
            num_proprio_frames=self.num_proprio_frames,
            proprio_dim=self.proprio_dim,
            critic_obs_dim=self.critic_obs_dim,
            height_map_dim=self.height_map_dim,
            foot_clearance_dim=self.foot_clearance_dim,
            action_dim=self.num_actions,
            device=self.device,
        )

    def _create_algorithm(self, cfg: dict) -> PIEPPO:
        """创建PPO算法"""
        alg_cfg = cfg.get("algorithm", {})

        return PIEPPO(
            actor_critic=self.actor_critic,
            storage=self.storage,
            learning_rate=alg_cfg.get("learning_rate", 1e-3),
            num_learning_epochs=alg_cfg.get("num_learning_epochs", 5),
            num_mini_batches=alg_cfg.get("num_mini_batches", 4),
            clip_param=alg_cfg.get("clip_param", 0.2),
            value_loss_coef=alg_cfg.get("value_loss_coef", 1.0),
            entropy_coef=alg_cfg.get("entropy_coef", 0.01),
            gamma=alg_cfg.get("gamma", 0.99),
            lam=alg_cfg.get("lam", 0.95),
            max_grad_norm=alg_cfg.get("max_grad_norm", 1.0),
            device=self.device,
        )

    def _init_history_buffers(self):
        """初始化历史缓冲区"""
        # 深度图历史 [num_envs, H1, H, W]
        self.depth_buffer = torch.zeros(
            self.num_envs, self.num_depth_frames,
            self.depth_height, self.depth_width,
            device=self.device,
        )

        # 本体感受历史 [num_envs, H2, proprio_dim]
        self.proprio_buffer = torch.zeros(
            self.num_envs, self.num_proprio_frames, self.proprio_dim,
            device=self.device,
        )

        # Episode统计
        self.episode_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device)

    def _update_depth_buffer(self, new_depth: torch.Tensor):
        """更新深度图缓冲区 (FIFO)

        Args:
            new_depth: [num_envs, H, W] 新的深度图
        """
        # 左移并添加新帧
        self.depth_buffer[:, :-1] = self.depth_buffer[:, 1:].clone()
        self.depth_buffer[:, -1] = new_depth

    def _update_proprio_buffer(self, new_proprio: torch.Tensor):
        """更新本体感受缓冲区 (FIFO)

        Args:
            new_proprio: [num_envs, proprio_dim] 新的本体感受
        """
        # 左移并添加新帧
        self.proprio_buffer[:, :-1] = self.proprio_buffer[:, 1:].clone()
        self.proprio_buffer[:, -1] = new_proprio

    def _reset_buffers_for_envs(self, env_ids: torch.Tensor):
        """为特定环境重置缓冲区

        Args:
            env_ids: 需要重置的环境索引
        """
        self.depth_buffer[env_ids] = 0.0
        self.proprio_buffer[env_ids] = 0.0
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_length[env_ids] = 0.0

    def _get_observations(self) -> Tuple[torch.Tensor, ...]:
        """从环境获取观测

        Returns:
            depth_image: [num_envs, H, W] 当前深度图
            current_proprio: [num_envs, proprio_dim] 当前本体感受
            critic_obs: [num_envs, critic_dim] Critic观测
            gt_velocity: [num_envs, 3] 真实速度
            gt_foot_clearance: [num_envs, 4] 真实足部间隙
            gt_height_map: [num_envs, height_map_dim] 真实高度图
        """
        # 获取unwrapped环境以访问observation_manager
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        obs_dict = unwrapped_env.observation_manager.compute()

        # 获取policy观测（本体感受）
        if "policy" in obs_dict:
            current_proprio = obs_dict["policy"]
        else:
            current_proprio = torch.zeros(
                self.num_envs, self.proprio_dim, device=self.device
            )

        # 获取深度图观测 - 从传感器直接获取
        depth_image = self._get_depth_from_sensor()

        # 获取Critic观测
        critic_obs_list = []
        for group_name in ["critic", "height_scan_group"]:
            if group_name in obs_dict:
                critic_obs_list.append(obs_dict[group_name])
        if critic_obs_list:
            critic_obs = torch.cat(critic_obs_list, dim=-1)
        else:
            critic_obs = torch.zeros(
                self.num_envs, self.critic_obs_dim, device=self.device
            )

        # 获取监督信号
        # 真实速度（从Critic观测中提取前3维，通常是base_lin_vel）
        gt_velocity = critic_obs[:, :3] if critic_obs.shape[-1] >= 3 else \
                      torch.zeros(self.num_envs, 3, device=self.device)

        # 足部间隙（需要从环境计算）
        gt_foot_clearance = self._compute_foot_clearance()

        # 高度图（从Critic观测或传感器获取）
        if "height_scan_group" in obs_dict:
            gt_height_map = obs_dict["height_scan_group"]
        else:
            gt_height_map = torch.zeros(
                self.num_envs, self.height_map_dim, device=self.device
            )

        return (
            depth_image,
            current_proprio,
            critic_obs,
            gt_velocity,
            gt_foot_clearance,
            gt_height_map,
        )

    def _get_depth_from_sensor(self) -> torch.Tensor:
        """从深度相机传感器获取深度图

        Returns:
            depth_image: [num_envs, H, W] 处理后的深度图
        """
        try:
            # 尝试从环境的scene获取depth_camera传感器
            unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(unwrapped_env, 'scene') and "depth_camera" in unwrapped_env.scene.sensors:
                camera = unwrapped_env.scene.sensors["depth_camera"]
                depth_data = camera.data.output["distance_to_image_plane"]

                # 获取图像尺寸
                img_height = camera.cfg.pattern_cfg.height
                img_width = camera.cfg.pattern_cfg.width

                # reshape 为图像格式
                if len(depth_data.shape) == 2:
                    # [num_envs, H*W] -> [num_envs, W, H] -> [num_envs, H, W]
                    depth_image = depth_data.reshape(self.num_envs, img_width, img_height)
                    depth_image = depth_image.permute(0, 2, 1)  # 转置为 [num_envs, H, W]
                else:
                    depth_image = depth_data.squeeze(-1)

                # 处理无效值
                depth_image = torch.nan_to_num(depth_image, nan=3.0, posinf=3.0, neginf=0.3)

                # 裁剪和归一化 (clip_range = 0.3, 3.0)
                min_depth, max_depth = 0.3, 3.0
                depth_image = torch.clamp(depth_image, min_depth, max_depth)
                depth_image = (depth_image - min_depth) / (max_depth - min_depth) - 0.5

                # 缩放到目标尺寸 (self.depth_height, self.depth_width)
                if depth_image.shape[1:] != (self.depth_height, self.depth_width):
                    depth_image = torch.nn.functional.interpolate(
                        depth_image.unsqueeze(1),  # [N, 1, H, W]
                        size=(self.depth_height, self.depth_width),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)  # [N, H, W]

                return depth_image

        except Exception as e:
            # 调试信息
            if self.current_iteration % 100 == 0:
                print(f"[PIE Runner] Could not get depth from sensor: {e}")

        # 返回零张量
        return torch.zeros(
            self.num_envs, self.depth_height, self.depth_width,
            device=self.device,
        )

    def _compute_foot_clearance(self) -> torch.Tensor:
        """计算足部间隙

        Returns:
            foot_clearance: [num_envs, 4] 4个足部的离地高度
        """
        # 简化实现：返回零张量
        # 实际应该从环境的robot asset计算
        return torch.zeros(
            self.num_envs, self.foot_clearance_dim, device=self.device
        )

    def learn(
        self,
        num_learning_iterations: Optional[int] = None,
        init_at_random_ep_len: bool = True,
    ) -> Tuple[PIEActorCritic, Dict]:
        """训练循环

        Args:
            num_learning_iterations: 训练迭代次数（覆盖配置）
            init_at_random_ep_len: 是否在随机episode长度处初始化

        Returns:
            actor_critic: 训练后的模型
            infos: 训练信息
        """
        # 使用参数覆盖配置
        if num_learning_iterations is not None:
            self.max_iterations = num_learning_iterations
        # 重置环境
        obs, _ = self.env.reset()
        self._init_history_buffers()

        # 获取初始观测
        depth_image, current_proprio, critic_obs, \
            gt_velocity, gt_foot_clearance, gt_height_map = self._get_observations()

        # 更新历史缓冲区
        self._update_depth_buffer(depth_image)
        self._update_proprio_buffer(current_proprio)

        start_time = time.time()

        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            iter_start = time.time()
            collection_start = time.time()

            # 收集轨迹
            with torch.no_grad():
                for step in range(self.num_steps_per_env):
                    # 准备输入
                    depth_images = self.depth_buffer.clone()
                    proprio_history = self.proprio_buffer.reshape(
                        self.num_envs, -1
                    )

                    # 采样动作
                    actions, values, actions_log_prob, action_mean, action_sigma = \
                        self.alg.act(
                            depth_images, proprio_history, current_proprio, critic_obs
                        )

                    # 与环境交互
                    obs, rewards, dones, infos = self.env.step(actions)

                    # 获取新观测
                    next_depth, next_proprio, next_critic_obs, \
                        next_gt_velocity, next_gt_fc, next_gt_map = self._get_observations()

                    # 存储转移
                    transition = PIETransition(
                        depth_images=depth_images,
                        proprio_history=proprio_history,
                        current_proprio=current_proprio,
                        critic_observations=critic_obs,
                        gt_velocity=gt_velocity,
                        gt_foot_clearance=gt_foot_clearance,
                        gt_height_map=gt_height_map,
                        next_proprio=next_proprio,
                        actions=actions,
                        rewards=rewards,
                        dones=dones.float(),
                        values=values,
                        actions_log_prob=actions_log_prob,
                        action_mean=action_mean,
                        action_sigma=action_sigma,
                    )
                    self.storage.add_transitions(transition)

                    # 更新统计
                    self.episode_reward_sum += rewards
                    self.episode_length += 1
                    self.total_timesteps += self.num_envs

                    # 收集episode info用于详细日志（每一步检查）
                    if "log" in infos and len(infos["log"]) > 0:
                        self.ep_infos.append(infos["log"])

                    # 处理完成的episode
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids) > 0:
                        for idx in done_ids:
                            self.episode_rewards.append(
                                self.episode_reward_sum[idx].item()
                            )
                            self.episode_lengths.append(
                                self.episode_length[idx].item()
                            )

                        # 重置缓冲区和Estimator隐状态
                        self._reset_buffers_for_envs(done_ids)
                        self.alg.process_env_reset(done_ids)

                    # 更新历史缓冲区
                    self._update_depth_buffer(next_depth)
                    self._update_proprio_buffer(next_proprio)

                    # 更新当前观测
                    current_proprio = next_proprio
                    critic_obs = next_critic_obs
                    gt_velocity = next_gt_velocity
                    gt_foot_clearance = next_gt_fc
                    gt_height_map = next_gt_map

                # 计算最后一步的价值（用于GAE）
                _, last_values, _, _, _ = self.alg.act(
                    self.depth_buffer,
                    self.proprio_buffer.reshape(self.num_envs, -1),
                    current_proprio,
                    critic_obs,
                )
                last_values = last_values.unsqueeze(-1)

            collection_time = time.time() - collection_start

            # 计算回报
            self.storage.compute_returns(
                last_values,
                gamma=self.alg.gamma,
                lam=self.alg.lam,
            )

            # PPO更新
            learn_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start

            # 清空存储
            self.storage.clear()

            # 分离GRU隐状态
            self.actor_critic.estimator.detach_hidden_state()

            # 计算迭代时间
            iteration_time = time.time() - iter_start
            self.tot_time += iteration_time

            # 日志
            if self.writer is not None:
                self._log_metrics(iteration, loss_dict)

            # 保存检查点
            if iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))

            # 详细日志输出
            if iteration % 10 == 0:
                self._print_log(
                    iteration=iteration,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    iteration_time=iteration_time,
                    loss_dict=loss_dict,
                )
                # 清空ep_infos
                self.ep_infos = []

        # 保存最终模型
        self.save(os.path.join(self.log_dir, "model_final.pt"))

        return self.actor_critic, {}

    def _log_metrics(self, iteration: int, loss_dict: Dict[str, float]):
        """记录TensorBoard指标"""
        if self.writer is None:
            return

        # 损失
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Loss/{k}", v, iteration)

        # Episode统计
        if len(self.episode_rewards) > 0:
            self.writer.add_scalar(
                "Train/mean_reward",
                sum(self.episode_rewards) / len(self.episode_rewards),
                iteration,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                sum(self.episode_lengths) / len(self.episode_lengths),
                iteration,
            )

        # 学习率
        self.writer.add_scalar(
            "Train/learning_rate",
            self.alg.learning_rate,
            iteration,
        )

    def _print_log(
        self,
        iteration: int,
        collection_time: float,
        learn_time: float,
        iteration_time: float,
        loss_dict: Dict[str, float],
    ):
        """打印详细的训练日志"""
        width = 80
        pad = 35

        # 计算FPS
        fps = self.num_envs * self.num_steps_per_env / (collection_time + learn_time)

        # 获取action std
        mean_std = self.actor_critic.std.mean().item()

        # 标题
        title_str = f" \033[1m Learning iteration {iteration}/{self.max_iterations} \033[0m "

        # 构建日志字符串
        if len(self.episode_rewards) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width + 8, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {loss_dict.get('ppo/value_loss', 0):.4f}\n"
                f"{'Surrogate loss:':>{pad}} {loss_dict.get('ppo/policy_loss', 0):.4f}\n"
                f"{'Entropy:':>{pad}} {loss_dict.get('ppo/entropy_loss', 0):.4f}\n"
            )

            # PIE Estimator损失
            if 'estimator/total_loss' in loss_dict:
                log_string += f"{'Estimator total loss:':>{pad}} {loss_dict.get('estimator/total_loss', 0):.4f}\n"
            if 'estimator/vel_loss' in loss_dict:
                log_string += f"{'Velocity loss:':>{pad}} {loss_dict.get('estimator/vel_loss', 0):.4f}\n"
            if 'estimator/fc_loss' in loss_dict:
                log_string += f"{'Foot clearance loss:':>{pad}} {loss_dict.get('estimator/fc_loss', 0):.4f}\n"
            if 'estimator/state_loss' in loss_dict:
                log_string += f"{'State prediction loss:':>{pad}} {loss_dict.get('estimator/state_loss', 0):.4f}\n"
            if 'estimator/map_loss' in loss_dict:
                log_string += f"{'Height map loss:':>{pad}} {loss_dict.get('estimator/map_loss', 0):.4f}\n"
            if 'estimator/kl_loss' in loss_dict:
                log_string += f"{'KL loss:':>{pad}} {loss_dict.get('estimator/kl_loss', 0):.4f}\n"

            log_string += (
                f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
                f"{'Mean reward:':>{pad}} {statistics.mean(self.episode_rewards):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(self.episode_lengths):.2f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{title_str.center(width + 8, ' ')}\n\n"
                f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
                f"{'Value function loss:':>{pad}} {loss_dict.get('ppo/value_loss', 0):.4f}\n"
                f"{'Surrogate loss:':>{pad}} {loss_dict.get('ppo/policy_loss', 0):.4f}\n"
                f"{'Entropy:':>{pad}} {loss_dict.get('ppo/entropy_loss', 0):.4f}\n"
                f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
            )

        # Episode奖励分解
        ep_string = ""
        if len(self.ep_infos) > 0:
            # 聚合所有ep_infos的值
            aggregated = {}
            for info in self.ep_infos:
                if isinstance(info, dict):
                    for key, val in info.items():
                        if key not in aggregated:
                            aggregated[key] = []
                        if isinstance(val, torch.Tensor):
                            aggregated[key].append(val.mean().item())
                        elif isinstance(val, (int, float)):
                            aggregated[key].append(val)

            # 按类别输出
            for prefix in ["Episode_Reward", "Episode_Termination", "Curriculum", "Metrics"]:
                for key in sorted(aggregated.keys()):
                    if key.startswith(prefix) and len(aggregated[key]) > 0:
                        mean_val = statistics.mean(aggregated[key])
                        ep_string += f" {key}: {mean_val:.4f}\n"

        log_string += ep_string

        # 时间统计
        remaining_iters = self.max_iterations - iteration - 1
        completed_iters = iteration - self.current_learning_iteration + 1

        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.total_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )

        if completed_iters > 0 and remaining_iters > 0:
            avg_time_per_iter = self.tot_time / completed_iters
            eta = avg_time_per_iter * remaining_iters
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"

        print(log_string)

    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict":
                self.actor_critic.estimator.optimizer.state_dict(),
            "iteration": self.current_iteration,
            "total_timesteps": self.total_timesteps,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "estimator_optimizer_state_dict" in checkpoint:
            self.actor_critic.estimator.optimizer.load_state_dict(
                checkpoint["estimator_optimizer_state_dict"]
            )
        self.current_iteration = checkpoint.get("iteration", 0)
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        print(f"Loaded model from {path}")

    def add_git_repo_to_log(self, file_path: str):
        """记录Git仓库信息

        Args:
            file_path: 脚本文件路径
        """
        if self.log_dir is None:
            return

        try:
            import subprocess
            repo_path = os.path.dirname(os.path.abspath(file_path))

            # 获取git信息
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path, stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            git_diff = subprocess.check_output(
                ["git", "diff", "--stat"],
                cwd=repo_path, stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            # 保存到文件
            with open(os.path.join(self.log_dir, "git_info.txt"), "w") as f:
                f.write(f"Git Hash: {git_hash}\n")
                f.write(f"Git Diff:\n{git_diff}\n")

        except Exception as e:
            print(f"[WARNING] Could not save git info: {e}")

    def get_inference_policy(self, device: Optional[str] = None):
        """获取推理策略

        Args:
            device: 目标设备

        Returns:
            policy: 推理策略函数
        """
        if device is not None:
            self.actor_critic.to(device)

        self.actor_critic.eval()

        # 初始化推理时的历史缓冲区
        self._init_history_buffers()

        def policy(obs: torch.Tensor) -> torch.Tensor:
            """推理策略

            Args:
                obs: [num_envs, obs_dim] 观测（本体感受）

            Returns:
                actions: [num_envs, action_dim] 动作
            """
            with torch.no_grad():
                # 更新本体感受缓冲区
                self._update_proprio_buffer(obs)

                # 准备输入
                depth_images = self.depth_buffer.clone()
                proprio_history = self.proprio_buffer.reshape(self.num_envs, -1)
                current_proprio = obs

                # 获取动作
                actions = self.actor_critic.act_inference(
                    depth_images, proprio_history, current_proprio
                )

                return actions

        return policy
