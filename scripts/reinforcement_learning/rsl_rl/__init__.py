"""
RSL-RL integration for robot_lab

This module provides custom runners and wrappers for training with velocity estimation.
"""

from .thunder_runner import ThunderRunner, create_thunder_runner
from .velocity_estimator_wrapper import (
    VelocityEstimatorWrapper,
    VelocityEstimatorConfig,
    create_velocity_estimator_wrapper,
)

__all__ = [
    "ThunderRunner",
    "create_thunder_runner",
    "VelocityEstimatorWrapper",
    "VelocityEstimatorConfig",
    "create_velocity_estimator_wrapper",
]

