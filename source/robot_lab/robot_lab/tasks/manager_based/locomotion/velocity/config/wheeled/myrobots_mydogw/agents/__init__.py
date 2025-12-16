# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MyDog agent configurations."""

from .rsl_rl_ppo_cfg import (
    MyDogFlatPPORunnerCfg,
    MyDogHistFlatPPORunnerCfg,
    MyDogHistRoughPPORunnerCfg,
    MyDogRoughPPORunnerCfg,
)

from .pie_ppo_cfg import (
    PIERunnerCfg,
    PIEEstimatorCfg,
    PIEPolicyCfg,
    PIEAlgorithmCfg,
    MyDogPIEFlatRunnerCfg,
    MyDogPIERoughRunnerCfg,
    MyDogPIEParkourRunnerCfg,
)

__all__ = [
    # 标准PPO配置
    "MyDogRoughPPORunnerCfg",
    "MyDogFlatPPORunnerCfg",
    # HIM风格PPO配置
    "MyDogHistRoughPPORunnerCfg",
    "MyDogHistFlatPPORunnerCfg",
    # PIE配置
    "PIERunnerCfg",
    "PIEEstimatorCfg",
    "PIEPolicyCfg",
    "PIEAlgorithmCfg",
    "MyDogPIEFlatRunnerCfg",
    "MyDogPIERoughRunnerCfg",
    "MyDogPIEParkourRunnerCfg",
]
