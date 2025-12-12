"""
Velocity Estimator Wrapper for Isaac Lab

Wrapper that integrates Velocity Estimator with Isaac Lab environments.
Handles observation history, estimator training, and provides predicted features to policy.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Import networks from robot_lab
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from robot_lab.networks import (
    VelocityEstimatorBarlowTwins,
    ImplicitEncoder,
    ObservationHistoryBuffer,
)


@dataclass
class VelocityEstimatorConfig:
    """Configuration for velocity estimator."""
    
    obs_dim: int
    history_len: int = 10
    hidden_dim: int = 128
    latent_dim: int = 12
    encoder_type: str = "gru"  # "gru", "transformer", "conv1d"
    use_barlow_twins: bool = True
    barlow_proj_dim: int = 64
    learning_rate: float = 1e-3
    velocity_weight: float = 1.0
    latent_weight: float = 1.0
    barlow_weight: float = 0.1
    lambda_off_diag: float = 5e-3
    device: str = "cuda:0"


class VelocityEstimatorWrapper:
    """
    Wrapper for Velocity Estimator integration with RL training.
    
    This wrapper:
    1. Maintains observation history buffer
    2. Trains velocity estimator using Barlow Twins self-supervised learning
    3. Provides predicted velocity and latent features to policy
    4. Logs training metrics
    
    The estimator learns to predict:
    - Base linear velocity (3D)
    - Latent environment features (12D): external forces, friction, contact, etc.
    """
    
    def __init__(self, env, config: VelocityEstimatorConfig):
        """
        Initialize velocity estimator wrapper.
        
        Args:
            env: Isaac Lab environment instance
            config: VelocityEstimatorConfig
        """
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        
        # Get observation dimensions
        obs = env.get_observations()
        if isinstance(obs, dict):
            policy_obs = obs['policy']
        else:
            policy_obs = obs
        
        self.num_envs = env.num_envs
        self.single_obs_dim = policy_obs.shape[1]
        
        print("\n" + "=" * 80)
        print("🚀 Velocity Estimator Wrapper Initialization")
        print("=" * 80)
        print(f"  Environment:      {env.__class__.__name__}")
        print(f"  Num Envs:         {self.num_envs}")
        print(f"  Obs Dimension:    {self.single_obs_dim}")
        print(f"  History Length:   {config.history_len} frames (≈{config.history_len * 0.04:.2f}s at 25Hz)")
        print(f"  Latent Dimension: {config.latent_dim}")
        print(f"  Encoder Type:     {config.encoder_type.upper()}")
        print(f"  Use Barlow Twins: {config.use_barlow_twins}")
        print(f"  Device:           {config.device}")
        
        # Create velocity estimator
        self.velocity_estimator = VelocityEstimatorBarlowTwins(
            obs_dim=self.single_obs_dim,
            history_len=config.history_len,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            encoder_type=config.encoder_type,
            use_barlow_twins=config.use_barlow_twins,
            barlow_proj_dim=config.barlow_proj_dim,
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.velocity_estimator.parameters())
        print(f"  Parameters:       {num_params:,}")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.velocity_estimator.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        
        # Create gradient scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Create observation history buffer
        self.obs_buffer = ObservationHistoryBuffer(
            num_envs=self.num_envs,
            history_len=config.history_len,
            obs_dim=self.single_obs_dim,
            device=self.device,
        )
        
        # Create implicit encoder (optional, for privileged feature encoding)
        self.implicit_encoder = None
        # You can create and train implicit encoder if needed
        
        # Training statistics
        self.training_step = 0
        self.metrics_history = {
            'velocity_error': [],
            'latent_loss': [],
            'barlow_loss': [],
        }
        
        # Cached predictions (updated each step)
        self.last_predicted_velocity = None
        self.last_predicted_latent = None
        
        print("=" * 80)
        print("✅ Velocity Estimator Ready!")
        print("=" * 80)
    
    def update_obs_buffer(self, obs: torch.Tensor):
        """
        Update observation history buffer.
        
        Args:
            obs: Current proprioceptive observations [num_envs, obs_dim]
        """
        self.obs_buffer.insert(obs.to(self.device))
    
    def get_observation_history(self) -> torch.Tensor:
        """
        Get observation history from buffer.
        
        Returns:
            history: [num_envs, history_len, obs_dim]
        """
        return self.obs_buffer.get_history()
    
    def predict(self, obs_history: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict velocity and latent features from observation history.
        
        Args:
            obs_history: Optional observation history [num_envs, history_len, obs_dim]
                        If None, uses buffer's history
        
        Returns:
            velocity: Predicted velocity [num_envs, 3]
            latent: Predicted latent [num_envs, latent_dim]
        """
        if obs_history is None:
            obs_history = self.obs_buffer.get_history()
        
        with torch.no_grad():
            velocity, latent = self.velocity_estimator(obs_history)
        
        # Cache predictions
        self.last_predicted_velocity = velocity
        self.last_predicted_latent = latent
        
        return velocity, latent
    
    def get_actor_input(self) -> torch.Tensor:
        """
        Get actor input (proprioception + predicted velocity + predicted latent).
        
        Returns:
            actor_input: [num_envs, obs_dim + 3 + latent_dim]
        """
        # Get current proprioceptive observation
        obs = self.env.get_observations()
        if isinstance(obs, dict):
            proprio_obs = obs['policy']
        else:
            proprio_obs = obs
        
        # Get predictions
        if self.obs_buffer.is_filled:
            velocity, latent = self.predict()
        else:
            # If buffer not filled, use zeros
            velocity = torch.zeros(self.num_envs, 3, device=self.device)
            latent = torch.zeros(self.num_envs, self.config.latent_dim, device=self.device)
        
        # Concatenate
        actor_input = torch.cat([proprio_obs, velocity, latent], dim=-1)
        
        return actor_input
    
    def train_step(self) -> dict:
        """
        Train velocity estimator for one step.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if not self.obs_buffer.is_filled:
            return {}
        
        # Get observation history
        obs_history = self.obs_buffer.get_history()
        
        # Get ground truth velocity
        if hasattr(self.env, 'robot'):
            gt_velocity = self.env.robot.data.root_lin_vel_b[:, :3]
        elif hasattr(self.env, 'scene') and 'robot' in self.env.scene:
            gt_velocity = self.env.scene["robot"].data.root_lin_vel_b[:, :3]
        else:
            raise RuntimeError("Cannot find robot to get ground truth velocity")
        
        # Get ground truth latent (if using implicit encoder)
        # For now, use placeholder or privileged features directly
        gt_latent = self._get_privileged_features()
        
        # Train step with AMP
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Compute full loss (velocity + latent + barlow twins)
            total_loss, loss_dict = self.velocity_estimator.compute_full_loss(
                obs_history=obs_history,
                gt_velocity=gt_velocity,
                gt_latent=gt_latent,
                velocity_weight=self.config.velocity_weight,
                latent_weight=self.config.latent_weight,
                barlow_weight=self.config.barlow_weight,
                lambda_off_diag=self.config.lambda_off_diag,
            )
        
        # Backward with gradient scaling and clipping
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.velocity_estimator.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Compute additional metrics
        with torch.no_grad():
            pred_velocity, pred_latent = self.velocity_estimator(obs_history)
            velocity_error = (pred_velocity - gt_velocity).abs().mean().item() * 1000  # Convert to mm/s
            direction_error = 1.0 - F.cosine_similarity(pred_velocity, gt_velocity, dim=-1).mean().item()
        
        # Update metrics
        metrics = {
            'velocity_error': velocity_error,
            'velocity_direction_error': direction_error,
            'latent_loss': loss_dict.get('latent_loss', 0.0),
            'velocity_loss': loss_dict.get('velocity_loss', 0.0),
            'total_loss': loss_dict.get('total_loss', 0.0),
        }
        
        if self.config.use_barlow_twins:
            metrics.update({
                'barlow_total': loss_dict.get('barlow_total', 0.0),
                'barlow_on_diag': loss_dict.get('barlow_on_diag', 0.0),
                'barlow_off_diag': loss_dict.get('barlow_off_diag', 0.0),
            })
        
        # Update history
        for key in ['velocity_error', 'latent_loss']:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
                if len(self.metrics_history[key]) > 100:
                    self.metrics_history[key] = self.metrics_history[key][-100:]
        
        self.training_step += 1
        
        return metrics
    
    def _get_privileged_features(self) -> torch.Tensor:
        """
        Get privileged features for latent supervision.
        
        Returns:
            privileged_features: [num_envs, latent_dim]
        """
        # Placeholder: You should implement this based on your environment
        # Example privileged features:
        # - External forces on robot
        # - Ground friction coefficient
        # - Contact forces
        # - COM velocity in world frame
        # - Terrain slope
        # etc.
        
        # For now, return zeros or extract from environment
        try:
            # Try to get from environment extras
            if hasattr(self.env, 'extras') and 'privileged_features' in self.env.extras:
                return self.env.extras['privileged_features']
        except:
            pass
        
        # Return zeros as fallback
        return torch.zeros(self.num_envs, self.config.latent_dim, device=self.device)
    
    def get_privileged_features(self) -> torch.Tensor:
        """Public method to get privileged features."""
        return self._get_privileged_features()
    
    def reset_buffer(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset observation buffer for specific environments.
        
        Args:
            env_ids: Environment IDs to reset (None = reset all)
        """
        self.obs_buffer.reset(env_ids)
    
    def update_iteration(self, iteration: int):
        """
        Update current training iteration (useful for warmup/scheduling).
        
        Args:
            iteration: Current training iteration
        """
        # Can implement learning rate scheduling here
        pass
    
    def save(self, path: str):
        """
        Save velocity estimator checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.velocity_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'training_step': self.training_step,
            'config': {
                'obs_dim': self.single_obs_dim,
                'history_len': self.config.history_len,
                'hidden_dim': self.config.hidden_dim,
                'latent_dim': self.config.latent_dim,
                'encoder_type': self.config.encoder_type,
                'use_barlow_twins': self.config.use_barlow_twins,
            },
        }
        torch.save(checkpoint, path)
        print(f"[VelocityEstimatorWrapper] Saved checkpoint to {path}")
    
    def load(self, path: str):
        """
        Load velocity estimator checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_estimator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"[VelocityEstimatorWrapper] Loaded checkpoint from {path} (step {self.training_step})")


def create_velocity_estimator_wrapper(
    env,
    obs_dim: int,
    history_len: int = 10,
    latent_dim: int = 12,
    encoder_type: str = "gru",
    use_barlow_twins: bool = True,
    learning_rate: float = 1e-3,
    device: str = "cuda:0",
) -> VelocityEstimatorWrapper:
    """
    Factory function to create VelocityEstimatorWrapper.
    
    Args:
        env: Isaac Lab environment
        obs_dim: Observation dimension
        history_len: History length (frames)
        latent_dim: Latent dimension
        encoder_type: Encoder type ("gru", "transformer", "conv1d")
        use_barlow_twins: Whether to use Barlow Twins
        learning_rate: Learning rate
        device: Device
    
    Returns:
        VelocityEstimatorWrapper instance
    """
    config = VelocityEstimatorConfig(
        obs_dim=obs_dim,
        history_len=history_len,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        use_barlow_twins=use_barlow_twins,
        learning_rate=learning_rate,
        device=device,
    )
    
    return VelocityEstimatorWrapper(env, config)

