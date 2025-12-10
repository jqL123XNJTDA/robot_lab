# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MyDog 轮腿机器人环境注册

已注册环境列表：
==============================================================================
标准 PPO 版本（无历史观测）：
- RobotLab-Isaac-Velocity-Flat-MyDog-v0     : 平坦地形
- RobotLab-Isaac-Velocity-Rough-MyDog-v0    : 粗糙地形
- RobotLab-Isaac-Velocity-Handstand-MyDog-v0: 倒立训练

HIM 版本（带 5 帧历史观测，配合 train_him.py 使用）：
- RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0  : 平坦地形
- RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 : 粗糙地形
==============================================================================

HIM 训练命令示例：
python scripts/reinforcement_learning/rsl_rl/train_him.py \\
    --task RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 \\
    --num_envs 4096 \\
    --headless
"""

import gymnasium as gym

from . import agents
from . import flat_env_cfg, rough_env_cfg


# ==============================================================================
# 标准 PPO 版本环境注册
# ==============================================================================

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MyDogFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyDogRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MyDogRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Handstand-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogHandstandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MyDogFlatTrainerCfg",
    },
)


# ==============================================================================
# HIM 版本环境注册（带 5 帧历史观测）
# ==============================================================================

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogHistFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogHistFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MyDogHistFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyDogHistRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogHistRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MyDogHistRoughTrainerCfg",
    },
)
