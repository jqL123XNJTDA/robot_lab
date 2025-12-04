# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**robot_lab** (v2.3.0) 是一个基于 Isaac Lab 2.3.0 的机器人强化学习扩展库。它为各种机器人平台（四足机器人、轮式机器人、人形机器人）提供模块化的强化学习训练框架，使用 Isaac Sim 中的 GPU 加速仿真。并在sim_m.py中使用mujuco进行模型的验证，调试pid等。


## 项目现在目的

模仿其他机器人环境配置等，导入我自己的四足轮腿机器人模型，配置资产，训练环境，以及agents,增加rewards，训练调试以满足可以上台阶的需求，并在sim_m.py进行mujucoo中进行调试pid，并验证效果
其中我自己的四足轮腿机器人模型地址：/home/liu/Desktop/robot_lab/source/robot_lab/data/Robots/myrobots/mydog
配置资产地址：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/assets/mydog.py
训练环境及reward策略：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/myrobots_mydogw/rough_env_cfg.py
ppo_cfg：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/myrobots_mydogw/agents/rsl_rl_ppo_cfg.py

## 代码规定
关键代码需要写中文注释

## 核心命令

### 安装与验证
```bash
# 安装扩展包
python -m pip install -e source/robot_lab

# 验证安装 - 列出所有可用环境
python scripts/tools/list_envs.py
```

### 训练与评估 (RSL-RL)
```bash
# 训练 (默认 4096 个并行环境)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Rough-<Robot>-v0 \
    --headless \
    --num_envs=4096

# 评估训练好的策略
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=RobotLab-Isaac-Velocity-Rough-<Robot>-v0 \
    --num_envs=64

# 从检查点恢复训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=<ENV_NAME> \
    --resume \
    --load_run run_folder_name \
    --checkpoint /PATH/TO/model.pt

# 播放特定检查点
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=<ENV_NAME> \
    --load_run run_folder_name \
    --checkpoint /PATH/TO/model.pt

# 单机器人键盘控制播放
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=<ENV_NAME> \
    --keyboard

# 录制视频 (需要 ffmpeg)
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=<ENV_NAME> \
    --video \
    --video_length 200
```

### HIM 框架训练（带历史观测）
```bash
# HIM Flat 地形预训练（推荐先在平坦地形训练）
python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0 \
    --num_envs 4096 \
    --headless

# HIM Rough 地形训练
python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 \
    --num_envs 4096 \
    --headless

# 播放/导出 HIM 策略
python scripts/reinforcement_learning/rsl_rl/play_him.py \
    --task RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 \
    --num_envs 64
```

### 多 GPU 训练
```bash
# 单节点多 GPU (--nproc_per_node 表示 GPU 数量)
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=<ENV_NAME> \
    --headless \
    --distributed

# 多节点多 GPU (主节点)
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:5555 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=<ENV_NAME> \
    --headless \
    --distributed

# 多节点多 GPU (工作节点)
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=ip_of_master_machine:5555 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=<ENV_NAME> \
    --headless \
    --distributed
```

### 工具脚本
```bash
# 查看 TensorBoard
tensorboard --logdir=logs

# 清理临时 USD 缓存文件
rm -rf /tmp/IsaacLab/usd_*
# 或使用工具脚本
python scripts/tools/clean_trash.py

# 转换 URDF 到 USD
python scripts/tools/convert_urdf.py

# 转换 MJCF 到 USD
python scripts/tools/convert_mjcf.py

# 代码格式化
pip install pre-commit
pre-commit run --all-files
```

### MuJoCo 仿真
```bash
# 使用训练好的策略在 MuJoCo 中运行
python sim_m.py \
    --model source/robot_lab/data/Robots/myrobots/mydog/mjcf/thunder2_v1.xml \
    --policy logs/rsl_rl/<experiment_name>/exported/policy.pt
```

## 架构概览

### 配置层次结构

项目使用三层配置继承系统：

1. **基础配置** (`velocity_env_cfg.py`)
   - 定义通用环境结构（场景、命令、动作、观测、事件、奖励、终止条件、课程学习）
   - 位置：`tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

2. **机器人特定 Rough 配置** (`rough_env_cfg.py`)
   - 继承基础配置并针对特定机器人定制
   - 使用地形生成器（多难度级别）
   - 包含高度扫描器用于地形感知
   - 启用完整领域随机化（质量、摩擦力等）
   - 位置：`config/<type>/<robot>/rough_env_cfg.py`

3. **机器人特定 Flat 配置** (`flat_env_cfg.py`)
   - 继承 Rough 配置
   - 覆盖地形类型为平面（`terrain_type = "plane"`）
   - 禁用高度扫描器
   - 适用于初始策略学习

这种继承模式允许在训练模式之间轻松切换。

### 目录结构关键组件

```
robot_lab/
├── source/robot_lab/robot_lab/
│   ├── assets/                    # 机器人定义 (ArticulationCfg)
│   │   ├── unitree.py            # Unitree 机器人配置
│   │   ├── mydog.py              # 自定义机器人配置
│   │   └── ...                   # 其他制造商
│   │
│   ├── tasks/manager_based/locomotion/velocity/
│   │   ├── velocity_env_cfg.py   # 基础环境配置
│   │   ├── mdp/                  # MDP 组件（观测、奖励、事件等）
│   │   └── config/               # 机器人特定配置
│   │       ├── quadruped/        # 四足机器人
│   │       ├── wheeled/          # 轮式机器人
│   │       ├── humanoid/         # 人形机器人
│   │       └── others/           # 特殊任务
│   │
│   └── data/Robots/              # 机器人网格/URDF/MJCF 文件
│       ├── unitree/
│       ├── myrobots/
│       └── ...
│
├── scripts/
│   ├── reinforcement_learning/
│   │   ├── rsl_rl/              # RSL-RL 训练（默认）+ HIM 框架
│   │   │   ├── train.py         # 标准 PPO 训练
│   │   │   ├── train_him.py     # HIM 框架训练
│   │   │   ├── play_him.py      # HIM 策略播放/导出
│   │   │   ├── algorithms/      # HIM PPO 算法
│   │   │   ├── modules/         # HIM Actor-Critic 和 Estimator
│   │   │   ├── runners/         # HIM Runner
│   │   │   ├── storage/         # HIM Rollout Storage
│   │   │   └── utils/           # 观测重排序等工具
│   │   ├── cusrl/               # CusRL 训练（实验性）
│   │   └── skrl/                # SKRL 训练（实验性）
│   └── tools/                   # 工具脚本
│
├── logs/                        # 训练日志和检查点
└── outputs/                     # Hydra 输出
```

### 机器人资产配置系统

每个机器人制造商有自己的 Python 文件（例如 `assets/unitree.py`、`assets/mydog.py`），定义：
- `ArticulationCfg`：描述如何生成和控制机器人
- URDF/MJCF 路径（指向 `data/Robots/`）
- 初始姿态和关节配置
- 执行器配置（刚度/阻尼参数）
- 关节组（例如轮式机器人的 "legs"、"wheels"）

**轮式机器人模式示例**（mydog）：
```python
MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(...),
    init_state=ArticulationCfg.InitialStateCfg(...),
    actuators={
        "legs": ImplicitActuatorCfg(stiffness=..., damping=...),    # 位置控制
        "wheels": ImplicitActuatorCfg(stiffness=..., damping=...),  # 速度控制
    }
)
```

### 环境注册系统

环境使用 Gym 注册表系统在 `__init__.py` 文件中注册：

命名约定：`RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v<Version>`

示例：
```python
gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)
```

### 观测-动作架构

**观测**分为两组：
- **Policy Group**：智能体的观测（带噪声/损坏）
  - 基座速度（线性/角速度）
  - 投影重力
  - 关节位置/速度
  - 前一个动作
  - 高度扫描（用于粗糙地形）
  - 速度命令

- **Critic Group**：价值函数观测（无噪声）
  - 与 policy 相同结构但无损坏

**动作**根据机器人类型定义：
- **四足机器人**：关节位置目标（缩放后）
- **轮式机器人**：混合动作
  - 腿部关节位置控制
  - 轮子关节速度控制
- **人形机器人**：关节位置目标（特殊处理）

### 奖励系统

模块化奖励系统，包含 70+ 个奖励项，按类别组织：
- 根部惩罚：姿态、高度、加速度
- 关节惩罚：力矩、速度、加速度、限制
- 动作惩罚：平滑性、速率限制
- 接触惩罚：不希望的接触、过度力
- 速度跟踪：指数跟踪奖励
- 步态奖励：足部空中时间、接触模式、对称性
- 特殊奖励：向上姿态、轮子行为

权重可设为 0 并通过 `disable_zero_weight_rewards()` 自动移除。

### 领域随机化事件

事件在不同模式触发：
- **Startup**：材料属性、质量、惯性、质心随机化
- **Reset**：关节状态、基座姿态、执行器增益、外力
- **Interval**：在回合期间推动机器人

这实现了鲁棒的 sim-to-real 迁移。

### MuJoCo 集成

项目包含 MuJoCo 支持：
- **MJCF 文件**：位于 `data/Robots/*/mjcf/`
- **仿真脚本**：`sim_m.py` 和 `run_mujoco.py` 运行独立 MuJoCo 仿真
  - 使用来自 Isaac Lab 的训练策略
  - 实现可配置增益的 PD 控制
  - 用于在不同物理引擎中测试策略

**关键差异**：MuJoCo 使用不同的关节排序和索引（dof_ids 映射）。

**MuJoCo 脚本关键参数**：
- `SIM_DT`：物理步长（默认 0.0002s，5kHz）
- `DECIMATION`：控制频率分频（默认 100，即 50Hz 控制频率）
- `RAMP_UP_TIME`：软启动时间（默认 1.5s）
- PD 增益：`kp_leg`、`kd_leg`（腿部）和 `kp_wheel`、`kd_wheel`（轮子）

---

## HIM 框架详解 (History-based Implicit Model)

### HIM 核心思想

HIM (History-based Implicit Model) 是一个专为四足/轮腿机器人设计的强化学习框架，源自 HIMLoco 论文。其核心思想是：

1. **历史观测编码**：使用多帧历史观测（默认 5 帧）来推断机器人当前的动态状态和环境特征
2. **速度估计**：通过 Estimator 网络从历史观测中估计机器人速度，替代不可靠的 IMU 积分
3. **隐式状态学习**：使用 SwAV 对比学习从历史观测中学习隐式状态表示（latent），编码地形、动态等信息
4. **非对称 Actor-Critic**：Actor 使用估计值，Critic 使用真实特权信息

### HIM 架构组件

```
HIM 训练框架
├── HIMActorCritic (modules/him_actor_critic.py)
│   ├── Estimator: 历史观测 → [velocity(3D), latent(16D)]
│   ├── Actor: [current_obs + vel + latent] → actions
│   └── Critic: privileged_obs → value
│
├── HIMEstimator (modules/him_estimator.py)
│   ├── Encoder: 展平历史观测 → [velocity, latent]
│   ├── Target: 当前观测 → latent (用于 SwAV 对比)
│   ├── Proto: 可学习聚类中心 (num_prototype=16)
│   └── Sinkhorn-Knopp: 最优传输软分配
│
├── HIMPPO (algorithms/him_ppo.py)
│   ├── PPO 策略更新 (Actor + Critic)
│   └── SwAV 对比学习 (Estimator)
│
├── HIMRolloutStorage (storage/him_rollout_storage.py)
│   ├── 存储历史观测、动作、奖励
│   └── 存储 next_critic_obs 用于速度监督
│
└── HIMOnPolicyRunner (runners/him_on_policy_runner.py)
    ├── 观测重排序 (Isaac Lab → HIM 格式)
    ├── 历史长度自动检测
    ├── 观测组分配 (policy vs critic)
    └── TensorBoard 日志 (含 estimation_loss, swap_loss)
```

### HIM 输入输出详解

#### 1. 观测空间 (Input)

**Policy 观测** - 每帧 57 维 × 5 帧 = 285 维：
```
观测顺序（Isaac Lab 格式 - 按变量分组）:
[ang_vel_t-4...t-0, gravity_t-4...t-0, cmd_t-4...t-0, 
 jpos_t-4...t-0, jvel_t-4...t-0, action_t-4...t-0]

观测顺序（HIM 格式 - 按时间步分组）:
[all_vars_t-4, all_vars_t-3, all_vars_t-2, all_vars_t-1, all_vars_t-0]

每帧结构 (57D):
├── base_ang_vel: 3D (缩放 0.25)
├── projected_gravity: 3D
├── velocity_commands: 3D
├── joint_pos: 16D (轮子位置置零)
├── joint_vel: 16D (缩放 0.05)
└── last_action: 16D
```

**Critic 观测** - 包含特权信息：
```
├── base_lin_vel: 3D × 5 = 15D (速度真值，用于监督 Estimator)
├── base_ang_vel: 3D × 5 = 15D
├── projected_gravity: 3D × 5 = 15D
├── velocity_commands: 3D × 5 = 15D
├── joint_pos: 16D × 5 = 80D
├── joint_vel: 16D × 5 = 80D
├── last_action: 16D × 5 = 80D
└── height_scan: 187D (仅 Rough 地形)
```

#### 2. Estimator 网络 (HIMEstimator)

**输入**：历史观测 `[batch, 285]` (5帧 × 57维)

**Encoder 网络**：
```python
Encoder: Linear(285 → 128) → ELU → Linear(128 → 64) → ELU → Linear(64 → 19)
输出: [velocity(3D), latent(16D)]
```

**Target 网络**（用于 SwAV）：
```python
Target: Linear(57 → 128) → ELU → Linear(128 → 64) → ELU → Linear(64 → 16)
输入: 当前帧观测 [batch, 57]
输出: latent [batch, 16]
```

**SwAV 对比学习**：
```python
# 1. 计算相似度得分
score_s = z_source @ prototype.T  # Encoder latent vs Prototypes
score_t = z_target @ prototype.T  # Target latent vs Prototypes

# 2. Sinkhorn-Knopp 生成软分配
q_s = sinkhorn(score_s)  # 均衡化分配
q_t = sinkhorn(score_t)

# 3. 交叉预测损失
swap_loss = -0.5 * (q_s * log_softmax(score_t/τ) + q_t * log_softmax(score_s/τ)).mean()

# 4. 速度估计损失（使用 t+1 时刻的真实速度作为监督）
estimation_loss = MSE(pred_vel, next_critic_obs[:, :3])
```

#### 3. Actor 网络

**输入**：`[current_obs(57D) + estimated_vel(3D) + latent(16D)] = 76D`

**网络结构**：
```python
Actor: Linear(76 → 512) → ELU → Linear(512 → 256) → ELU → Linear(256 → 128) → ELU → Linear(128 → 16)
输出: action_mean [batch, 16]
```

**动作分布**：
```python
distribution = Normal(action_mean, learned_std)
actions = distribution.sample()  # 训练时
actions = action_mean  # 推理时（确定性）
```

#### 4. Critic 网络

**输入**：Critic 观测（包含特权信息）`[batch, critic_dim]`

**网络结构**：
```python
Critic: Linear(critic_dim → 512) → ELU → Linear(512 → 256) → ELU → Linear(256 → 128) → ELU → Linear(128 → 1)
输出: value [batch, 1]
```

#### 5. 动作空间 (Output)

```
MyDog 动作空间 (16D):
├── 腿部关节位置 (12D) - 位置控制
│   ├── FR: hip, thigh, calf (缩放: 0.125, 0.25, 0.25)
│   ├── FL: hip, thigh, calf
│   ├── RR: hip, thigh, calf
│   └── RL: hip, thigh, calf
└── 轮子关节速度 (4D) - 速度控制
    └── FR, FL, RR, RL foot (缩放: 5.0)
```

### HIM 训练流程

```python
# 1. 收集轨迹
for step in range(num_steps_per_env):
    # 重排序观测 (Isaac Lab → HIM 格式)
    obs_him = reshape_isaac_to_him(obs_isaac, history_len=5, obs_dims=[3,3,3,16,16,16])
    
    # Actor 选择动作
    actions = actor_critic.act(obs_him)
    
    # 环境步进
    next_obs, rewards, dones, infos = env.step(actions)
    
    # 存储转换（包含 next_critic_obs 用于速度监督）
    storage.add(obs, actions, rewards, dones, next_critic_obs)

# 2. 计算 GAE 优势
storage.compute_returns(last_values, gamma=0.99, lam=0.95)

# 3. 更新网络
for epoch in range(num_epochs):
    for batch in storage.mini_batch_generator():
        # 更新 Estimator (SwAV + 速度估计)
        estimation_loss, swap_loss = estimator.update(obs_batch, next_critic_obs_batch)
        
        # 更新 Actor-Critic (PPO)
        actor_critic.act(obs_batch)
        surrogate_loss = compute_ppo_loss(...)
        value_loss = compute_value_loss(...)
        
        loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
        optimizer.step()
```

### HIM 关键配置

**环境配置** (`rough_env_cfg.py`)：
```python
@configclass
class MyDogHistObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        # 观测项定义...
        def __post_init__(self):
            self.history_length = 5  # 关键：启用历史观测
```

**PPO 配置** (`rsl_rl_ppo_cfg.py`)：
```python
@configclass
class MyDogHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 200  # 与 HIMLoco 论文一致
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"]
    }
    algorithm = RslRlPpoAlgorithmCfg(
        max_grad_norm=10.0,  # 与论文一致
        # ...
    )
```

### HIM 环境列表

| 环境 ID | 描述 | 训练脚本 |
|---------|------|----------|
| `RobotLab-Isaac-Velocity-Flat-MyDog-v0` | 标准 PPO，平坦地形 | `train.py` |
| `RobotLab-Isaac-Velocity-Rough-MyDog-v0` | 标准 PPO，粗糙地形 | `train.py` |
| `RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0` | HIM，平坦地形 | `train_him.py` |
| `RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0` | HIM，粗糙地形 | `train_him.py` |

### HIM 与标准 PPO 对比

| 特性 | 标准 PPO | HIM |
|------|----------|-----|
| 观测历史 | 1 帧 | 5 帧 |
| 速度输入 | 直接使用 IMU | Estimator 估计 |
| 隐式状态 | 无 | SwAV 学习的 latent |
| Actor 输入 | 原始观测 | 当前观测 + vel + latent |
| Critic | 相同观测 | 特权观测（含真实速度） |
| 训练脚本 | `train.py` | `train_him.py` |

---

## 添加新机器人的工作流程

1. **添加资产文件**
   - 将 URDF/网格文件添加到 `data/Robots/<manufacturer>/<robot>/`
   - 在 `assets/<manufacturer>.py` 中创建 `ArticulationCfg`

2. **创建任务配置**
   - 在 `config/<type>/<robot>/` 创建目录
   - 编写 `rough_env_cfg.py`（继承基础配置并定制）
   - 编写 `flat_env_cfg.py`（继承 rough 配置并覆盖地形）

3. **配置智能体**
   - 在 `agents/` 中创建 RL 智能体配置（`rsl_rl_ppo_cfg.py`、`cusrl_ppo_cfg.py`）

4. **注册环境**
   - 在 `__init__.py` 中使用 `gym.register()` 注册环境
   - 遵循命名约定

5. **验证**
   - 运行 `python scripts/tools/list_envs.py` 确认注册
   - 用小数量环境测试训练（`--num_envs 64`）

## 训练最佳实践

1. **先在平面地形训练**以实现初始学习
2. **使用课程学习**应对粗糙地形
3. **迭代调整奖励权重**
4. **监控 TensorBoard**：`tensorboard --logdir=logs`
5. **频繁在 play 模式测试**
6. **使用领域随机化**实现鲁棒性
7. **日志位置**：`logs/rsl_rl/<task_name>/<timestamp>/`
8. **检查点频率**：每 100 次迭代保存
9. **HIM 训练建议**：先用 Flat 环境验证 Estimator 收敛，再迁移到 Rough

## 重要架构模式

### SceneEntityCfg 模式
```python
SceneEntityCfg("robot", joint_names=".*_hip_joint")
```
在整个代码库中使用正则表达式匹配引用场景元素。

### Configclass 装饰器
```python
@configclass
class MyEnvCfg(BaseEnvCfg):
    ...
```
将 dataclass 转换为与 Isaac Lab 兼容的配置对象。

### 智能体配置分离
- 环境配置定义任务
- 智能体配置（在 `agents/` 中）定义学习算法
- 清晰的分离允许交换 RL 算法

### MDP 模块系统
- 所有 MDP 函数在 `mdp/` 子目录中
- 按以下组织：观测、奖励、事件、命令、课程
- 通过通配符导入：`from .mdp import *`

## 当前 MyDog 机器人配置

**MyDog** 是展示轮式腿式模式的自定义机器人：
- **资产**：`assets/mydog.py` - 定义带腿和轮子执行器的 `MYDOG_CFG`
- **URDF**：`data/Robots/myrobots/mydog/urdf/thunder_nohead.urdf`
- **MJCF**：`data/Robots/myrobots/mydog/mjcf/thunder2_v1.xml`
- **任务配置**：`config/wheeled/myrobots_mydogw/rough_env_cfg.py`
  - 混合动作空间：腿部位置控制，轮子速度控制
  - 特殊观测：排除轮子位置（无限旋转）
  - 轮子特定奖励和约束
  - 支持标准 PPO 和 HIM 框架
- **当前分支**：`feature/mydog`
- **最近训练**：`logs/rsl_rl/mydog_rough/` 和 `mydog_flat/`

## 关键依赖项

- Isaac Lab 2.3.0 / Isaac Sim 4.5/5.0/5.1
- Python 3.10+
- RSL-RL 3.0.1+（主要 RL 库）
- CusRL（实验性）
- pinocchio、pandas（用于 AMP/BeyondMimic）

## Docker 支持

```bash
# 构建镜像
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab

# 启动容器
docker compose --env-file .env.base --file docker-compose.yaml up -d

# 进入容器
docker exec --interactive --tty -e DISPLAY=${DISPLAY} robot-lab /bin/bash

# 关闭容器
docker compose --env-file .env.base --file docker-compose.yaml down
```

## 相关项目

- **rl_sar**：用于在 Gazebo 或真实机器人上运行策略
  - GitHub：https://github.com/fan-ziqi/rl_sar
- **讨论社区**：
  - GitHub Discussions：https://github.com/fan-ziqi/robot_lab/discussions
  - Discord：http://www.robotsfan.com/dc_robot_lab
