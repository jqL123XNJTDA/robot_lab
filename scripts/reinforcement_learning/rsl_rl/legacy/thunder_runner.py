"""
Thunder Runner - PPO Runner with Velocity Estimator Integration

Based on thunder2 project's ThunderRunner, adapted for Isaac Lab's robot_lab.

Features:
- Asymmetric Actor-Critic (Actor uses estimated velocity, Critic uses privileged info)
- Velocity Estimator training with Barlow Twins self-supervised learning
- Automatic observation dimension handling
- Episode tracking and statistics
"""

import time
import torch
import os
from typing import Optional

from rsl_rl.runners import OnPolicyRunner


class ObservationAdapter:
    """Adapter for handling Actor/Critic observation inputs."""
    
    def __init__(self, env, velocity_wrapper=None):
        """
        Args:
            env: Environment instance
            velocity_wrapper: Optional VelocityEstimatorWrapper instance
        """
        self.env = env
        self.velocity_wrapper = velocity_wrapper
        
        # Cache observation dimensions
        self._actor_obs_dim = None
        self._critic_obs_dim = None
    
    def get_actor_obs_dim(self):
        """Get Actor observation dimension."""
        if self._actor_obs_dim is None:
            actor_input = self.get_actor_critic_input()
            self._actor_obs_dim = actor_input['policy'].shape[1]
        return self._actor_obs_dim
    
    def get_critic_obs_dim(self):
        """Get Critic observation dimension."""
        if self._critic_obs_dim is None:
            actor_input = self.get_actor_critic_input()
            self._critic_obs_dim = actor_input['critic'].shape[1]
        return self._critic_obs_dim
    
    def _get_env_obs_for_policy(self):
        """Get Actor/Policy base environment observation."""
        obs = self.env.get_observations()
        if hasattr(obs, 'get'):
            return obs.get('policy')
        return obs
    
    def _get_env_obs_for_critic(self):
        """Get Critic environment observation (including privileged info)."""
        obs = self.env.get_observations()
        
        # Get base Critic observation
        if hasattr(obs, 'get'):
            critic_obs = obs.get('critic', obs.get('policy'))
        else:
            critic_obs = obs
        
        # If no velocity_wrapper, return environment observation directly
        if self.velocity_wrapper is None:
            return critic_obs
        
        # Add privileged information
        privileged_components = [critic_obs]
        
        # 1. True linear velocity (3D) - Most important privileged info
        try:
            if hasattr(self.env, 'robot'):
                gt_velocity = self.env.robot.data.root_lin_vel_b[:, :3]
            elif hasattr(self.env, 'scene') and 'robot' in self.env.scene:
                gt_velocity = self.env.scene["robot"].data.root_lin_vel_b[:, :3]
            else:
                gt_velocity = None
            
            if gt_velocity is not None:
                privileged_components.append(gt_velocity)
        except:
            pass
        
        # 2. Height Scan (terrain height scanning)
        try:
            if hasattr(self.env.scene, 'height_scanner') and self.env.scene.height_scanner is not None:
                height_data = self.env.scene.height_scanner.data
                if hasattr(height_data, 'pos_w') and hasattr(height_data, 'ray_hits_w'):
                    heights = height_data.pos_w[:, :, 2] - height_data.ray_hits_w[:, :, 2]
                    heights_flat = heights.flatten(start_dim=1)
                    heights_scaled = torch.clamp(heights_flat, -1.0, 1.0)
                    privileged_components.append(heights_scaled)
        except:
            pass
        
        # 3. Core privileged features (12D): external forces, friction, contact, etc.
        try:
            from robot_lab.networks import ImplicitEncoder
            # This should be provided by the wrapper
            if hasattr(self.velocity_wrapper, 'get_privileged_features'):
                priv_features = self.velocity_wrapper.get_privileged_features()
                privileged_components.append(priv_features)
        except:
            pass
        
        # Concatenate all observations
        if len(privileged_components) > 1:
            critic_obs_with_priv = torch.cat(privileged_components, dim=-1)
            return critic_obs_with_priv
        else:
            return critic_obs
    
    def get_actor_critic_input(self):
        """Get Actor and Critic inputs.
        
        Returns:
            dict: {'policy': actor_input, 'critic': critic_input}
        """
        if self.velocity_wrapper is not None:
            # With Velocity Estimator: Actor uses predicted velocity and latent
            actor_input = self.velocity_wrapper.get_actor_input()
        else:
            # Without Velocity Estimator: Use environment observation directly
            actor_input = self._get_env_obs_for_policy()
        
        # Critic always gets full privileged observation
        critic_input = self._get_env_obs_for_critic()
        
        return {
            'policy': actor_input,
            'critic': critic_input,
        }


class EpisodeTracker:
    """Episode statistics tracker."""
    
    def __init__(self, num_envs, device):
        """
        Args:
            num_envs: Number of environments
            device: Device (cpu/cuda)
        """
        self.num_envs = num_envs
        self.device = device
        
        # Episode statistics
        self.episode_length_buf = torch.zeros(num_envs, device=device)
        self.episode_reward_buf = torch.zeros(num_envs, device=device)
        
        # History for computing averages
        self.reward_history = []
        self.length_history = []
    
    def update(self, rewards, dones):
        """Update episode statistics.
        
        Args:
            rewards: Current step rewards [num_envs]
            dones: Current step done flags [num_envs]
        """
        self.episode_length_buf += 1
        self.episode_reward_buf += rewards
        
        # Detect completed episodes
        done_indices = dones.nonzero(as_tuple=False).flatten()
        
        if len(done_indices) > 0:
            # Record completed episodes
            for idx in done_indices:
                self.reward_history.append(self.episode_reward_buf[idx].item())
                self.length_history.append(self.episode_length_buf[idx].item())
            
            # Reset completed episodes
            self.episode_length_buf[done_indices] = 0
            self.episode_reward_buf[done_indices] = 0
    
    def get_statistics(self):
        """Get episode statistics.
        
        Returns:
            dict: Contains mean reward and mean length
        """
        if len(self.reward_history) == 0:
            return {
                'episode_reward_mean': 0.0,
                'episode_length_mean': 0.0,
            }
        
        # Keep only last 100 episodes
        if len(self.reward_history) > 100:
            self.reward_history = self.reward_history[-100:]
            self.length_history = self.length_history[-100:]
        
        return {
            'episode_reward_mean': sum(self.reward_history) / len(self.reward_history),
            'episode_length_mean': sum(self.length_history) / len(self.length_history),
        }


class ThunderRunner(OnPolicyRunner):
    """Thunder-specific PPO Runner with Velocity Estimator integration.
    
    Features:
    - Asymmetric Actor-Critic (Actor uses estimated velocity, Critic uses privileged info)
    - Optional Velocity Estimator wrapper integration
    - Automatic observation dimension handling
    - Episode tracking and statistics
    """
    
    def __init__(self, env, train_cfg, log_dir=None, device='cpu', velocity_wrapper=None):
        """
        Args:
            env: Environment instance
            train_cfg: Training configuration dict
            log_dir: Log directory
            device: Compute device
            velocity_wrapper: Optional VelocityEstimatorWrapper instance
        """
        # Save velocity wrapper reference
        self.velocity_wrapper = velocity_wrapper
        
        # Create observation adapter
        self.obs_adapter = ObservationAdapter(env, velocity_wrapper)
        
        # Create Episode tracker
        self.episode_tracker = EpisodeTracker(env.num_envs, device)
        
        # Print initialization info
        self._print_initialization_info()
        
        # Call parent initialization (creates Actor-Critic network)
        super().__init__(env, train_cfg, log_dir, device)
        
        print(f"[ThunderRunner] ✅ Initialization complete!\n")
    
    def _print_initialization_info(self):
        """Print initialization info."""
        actor_dim = self.obs_adapter.get_actor_obs_dim()
        critic_dim = self.obs_adapter.get_critic_obs_dim()
        
        print("=" * 70)
        print("ThunderRunner - Asymmetric Actor-Critic with Velocity Estimator")
        print("=" * 70)
        
        if self.velocity_wrapper is not None:
            print(f"📊 Network Configuration:")
            print(f"  Actor input:  {actor_dim}D")
            print(f"    ├─ Proprioception: {self.velocity_wrapper.single_obs_dim}D")
            print(f"    ├─ Predicted velocity: 3D")
            print(f"    └─ Predicted latent: {self.velocity_wrapper.config.latent_dim}D")
            print(f"  Critic input: {critic_dim}D (privileged)")
            print(f"\n🎓 Estimator Configuration:")
            print(f"  History length: {self.velocity_wrapper.config.history_len} frames")
            print(f"  Encoder type: {self.velocity_wrapper.config.encoder_type}")
            print(f"  Latent dim: {self.velocity_wrapper.config.latent_dim}D")
        else:
            print(f"📊 Network Configuration:")
            print(f"  Actor input:  {actor_dim}D")
            print(f"  Critic input: {critic_dim}D")
            print(f"\n⚠️  No Velocity Estimator (Standard PPO mode)")
        
        print("=" * 70)
    
    def get_inference_policy(self, device=None):
        """Override to return actor-only policy for deployment."""
        return self.alg.policy.actor
    
    def save(self, path, infos=None):
        """Save checkpoint, extends parent method to support velocity_estimator."""
        # Call parent method to save base checkpoint
        super().save(path, infos)
        
        # If has velocity_estimator, add to checkpoint
        if self.velocity_wrapper is not None:
            checkpoint = torch.load(path, map_location=self.device)
            checkpoint['velocity_estimator'] = {
                'model_state_dict': self.velocity_wrapper.velocity_estimator.state_dict(),
                'optimizer_state_dict': self.velocity_wrapper.optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"[ThunderRunner] Saved checkpoint with velocity_estimator to {path}")
    
    def load(self, path, load_optimizer=True, map_location=None):
        """Load checkpoint, extends parent method to support velocity_estimator."""
        # Call parent method to load base checkpoint
        infos = super().load(path, load_optimizer, map_location)
        
        # Load velocity estimator (if exists)
        if self.velocity_wrapper is not None:
            checkpoint = torch.load(path, weights_only=False, map_location=map_location)
            if 'velocity_estimator' in checkpoint:
                self.velocity_wrapper.velocity_estimator.load_state_dict(
                    checkpoint['velocity_estimator']['model_state_dict']
                )
                if load_optimizer:
                    self.velocity_wrapper.optimizer.load_state_dict(
                        checkpoint['velocity_estimator']['optimizer_state_dict']
                    )
                print(f"[ThunderRunner] Loaded velocity_estimator from {path}")
        
        return infos


def create_thunder_runner(env, train_cfg, log_dir, device, velocity_wrapper=None):
    """
    Factory function to create ThunderRunner.
    
    Args:
        env: Environment instance
        train_cfg: Training configuration dict
        log_dir: Log directory
        device: Compute device
        velocity_wrapper: Optional VelocityEstimatorWrapper
    
    Returns:
        ThunderRunner instance
    """
    return ThunderRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=device,
        velocity_wrapper=velocity_wrapper,
    )

