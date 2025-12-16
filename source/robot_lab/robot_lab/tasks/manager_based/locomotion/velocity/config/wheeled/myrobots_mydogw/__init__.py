# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents
from . import flat_env_cfg, rough_env_cfg, pie_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyDogRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Handstand-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogHandstandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogFlatPPORunnerCfg",
    },
)

# ==============================================================================
# HIM (History-based Implicit Model) 环境注册
# ==============================================================================

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogHistFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogHistFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyDogHistRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogHistRoughPPORunnerCfg",
    },
)

# ==============================================================================
# PIE (Parkour with Implicit-Explicit Learning Framework) 环境注册
# ==============================================================================

gym.register(
    id="RobotLab-Isaac-Velocity-PIE-Flat-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pie_env_cfg:MyDogPIEFlatEnvCfg",
        "pie_cfg_entry_point": f"{agents.__name__}.pie_ppo_cfg:MyDogPIEFlatRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-PIE-Rough-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pie_env_cfg:MyDogPIERoughEnvCfg",
        "pie_cfg_entry_point": f"{agents.__name__}.pie_ppo_cfg:MyDogPIERoughRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-PIE-Parkour-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pie_env_cfg:MyDogPIEParkourEnvCfg",
        "pie_cfg_entry_point": f"{agents.__name__}.pie_ppo_cfg:MyDogPIEParkourRunnerCfg",
    },
)
