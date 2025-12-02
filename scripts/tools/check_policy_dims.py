#!/usr/bin/env python3
"""Load an exported policy and report observation/action dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch


def _iter_linear_layers(module: torch.jit.ScriptModule) -> Iterable[torch.jit.ScriptModule]:
    """遍历 ScriptModule 中的线性层，便于自动提取权重维度。"""
    for _, child in module._modules.items():  # pylint: disable=protected-access
        if hasattr(child, "weight") and hasattr(child, "bias"):
            yield child


def _load_policy(path: Path) -> torch.jit.ScriptModule:
    """加载策略文件，优先使用 torch.jit.load，必要时退回常规 torch.load。"""
    try:
        return torch.jit.load(path, map_location="cpu")
    except RuntimeError:
        # torch.jit.load 失败时，有可能是普通 state_dict；强制 weights_only=False 再试一次。
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.jit.ScriptModule):
            return obj
        raise RuntimeError("给定 policy.pt 既不是 TorchScript 也不是可识别的策略对象")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect policy IO dimensions.")
    parser.add_argument("policy_path", type=Path, help="policy.pt 的绝对或相对路径")
    args = parser.parse_args()

    policy = _load_policy(args.policy_path)
    actor = getattr(policy, "actor", None)
    if actor is None:
        raise AttributeError("策略对象缺少 actor 子模块，无法推断维度")

    linear_layers = list(_iter_linear_layers(actor))
    if not linear_layers:
        raise RuntimeError("actor 中找不到线性层，无法推断维度")

    first_layer = linear_layers[0]
    last_layer = linear_layers[-1]
    observation_dim = first_layer.weight.shape[1]
    action_dim = last_layer.weight.shape[0]

    print(f"Loaded policy from: {args.policy_path}")
    print(f"Observation dim: {observation_dim}")
    print(f"Action dim: {action_dim}")


if __name__ == "__main__":
    main()
