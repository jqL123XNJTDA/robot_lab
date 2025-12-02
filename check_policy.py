#!/usr/bin/env python3
"""
检查 PyTorch 策略文件 (.pt) 的输入输出维度。

用法:
    python check_policy.py <policy_path>

示例:
    python check_policy.py logs/rsl_rl/helios_leg_flat/exported/policy.pt
"""

import argparse
import torch


def check_policy(policy_path: str):
    """加载并检查策略文件的结构和维度。"""

    print(f"\n{'='*60}")
    print(f"检查策略文件: {policy_path}")
    print(f"{'='*60}\n")

    # 加载模型
    try:
        model = torch.jit.load(policy_path)
        print("[✓] 成功加载 TorchScript 模型\n")
    except Exception as e:
        print(f"[!] TorchScript 加载失败: {e}")
        print("[*] 尝试使用 torch.load 加载...\n")
        try:
            checkpoint = torch.load(policy_path, map_location="cpu")
            print(f"[✓] 成功加载 checkpoint\n")
            print(f"Checkpoint 类型: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
            return
        except Exception as e2:
            print(f"[✗] 加载失败: {e2}")
            return

    # 打印模型结构
    print("【模型结构】")
    print("-" * 40)
    print(model)
    print()

    # 尝试获取输入维度
    print("【尝试推断输入/输出维度】")
    print("-" * 40)

    # 常见的观测维度列表（用于测试）
    test_dims = [25, 53, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12]

    for obs_dim in test_dims:
        try:
            # 创建测试输入 (batch_size=1)
            test_input = torch.zeros(1, obs_dim)
            output = model(test_input)

            if isinstance(output, tuple):
                output_shape = [o.shape for o in output]
            else:
                output_shape = output.shape

            print(f"\n[✓] 找到有效输入维度!")
            print(f"    输入维度 (obs_dim): {obs_dim}")
            print(f"    输出形状 (action):  {output_shape}")

            if not isinstance(output, tuple):
                print(f"    动作维度 (act_dim): {output.shape[-1]}")

            return obs_dim, output_shape

        except Exception:
            continue

    print("\n[!] 无法自动推断输入维度，请手动指定")
    print("    常见维度: helios_leg=25, mydog=53")

    return None, None


def check_with_dim(policy_path: str, obs_dim: int):
    """使用指定的观测维度检查策略。"""

    print(f"\n使用指定维度 obs_dim={obs_dim} 检查...")

    model = torch.jit.load(policy_path)
    test_input = torch.zeros(1, obs_dim)

    try:
        output = model(test_input)
        print(f"[✓] 输入维度: {obs_dim}")
        print(f"[✓] 输出形状: {output.shape}")
        print(f"[✓] 动作维度: {output.shape[-1]}")
    except Exception as e:
        print(f"[✗] 错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查策略文件的输入输出维度")
    parser.add_argument("policy_path", type=str, help="策略文件路径 (.pt)")
    parser.add_argument("--obs_dim", type=int, default=None, help="指定观测维度（可选）")

    args = parser.parse_args()

    if args.obs_dim:
        check_with_dim(args.policy_path, args.obs_dim)
    else:
        check_policy(args.policy_path)
