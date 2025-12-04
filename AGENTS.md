# 仓库指南

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

## 项目结构与模块组织

Isaac Lab 扩展位于 `source/robot_lab/`。核心 Python 包放在 `robot_lab/`，其中 `assets/` 定义机器人模型，`tasks/` 则包含 direct、manager_based 和 beyondmimic 环境。训练入口位于 `scripts/reinforcement_learning/`，`scripts/tools/` 包含 URDF/MJCF 转换与动作预处理等工具。机器人网格与 URDF/MJCF 文件保存在 `data/Robots/`。Docker 相关文件在 `docker/`，文档资产在 `docs/`，运行时产物写入 `logs/` 与 `outputs/`。

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
│   │   │   ├── train.py         # 标准 PPO 训练入口
│   │   │   ├── train_him.py     # HIM 框架训练入口
│   │   │   ├── play_him.py      # HIM 策略播放/导出
│   │   │   ├── algorithms/      # HIM PPO 算法实现
│   │   │   ├── modules/         # HIM Actor-Critic 和 Estimator 网络
│   │   │   ├── runners/         # HIM 训练 Runner
│   │   │   ├── storage/         # HIM Rollout Storage
│   │   │   └── utils/           # 观测重排序等工具函数
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

### 一、HIM 核心思想

HIM (History-based Implicit Model) 是一个专为四足/轮腿机器人设计的先进强化学习框架，源自 HIMLoco 论文。其核心创新在于：

#### 1.1 为什么需要历史观测？

传统 RL 方法直接使用 IMU 数据作为速度估计，存在以下问题：
- **积分漂移**：IMU 加速度积分会累积误差
- **噪声敏感**：高频振动导致测量不准确
- **状态不完整**：单帧观测无法捕捉动态信息

HIM 的解决方案：**使用多帧历史观测 + 神经网络估计**
```
历史观测序列 [t-4, t-3, t-2, t-1, t] → Estimator → [velocity, latent]
                                              ↓
                              比 IMU 积分更准确的速度估计
                              + 隐式编码的环境/动态信息
```

#### 1.2 SwAV 对比学习的作用

SwAV (Swapped Assignment between Views) 是一种自监督对比学习方法：

**核心思想**：让 Encoder 和 Target 网络学习一致的 latent 表示
```
Encoder(历史观测) → z_source → 分配到原型 → q_source
Target(当前观测)  → z_target → 分配到原型 → q_target

损失函数：用 q_source 预测 z_target，用 q_target 预测 z_source
```

**为什么有效**：
- 强制 latent 编码有意义的信息（不是随机噪声）
- 历史和当前观测的 latent 应该相似（同一状态）
- 不同状态的 latent 应该不同（区分能力）

#### 1.3 非对称 Actor-Critic 架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练时（仿真器中）                      │
├─────────────────────────────────────────────────────────┤
│  Actor: 历史观测 → Estimator → [obs + vel + latent]     │
│                                       ↓                 │
│                                  MLP → actions          │
│                                                         │
│  Critic: 特权观测（含真实速度）→ MLP → value             │
│          ↑                                              │
│     仅在训练时可用                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    部署时（真实机器人）                    │
├─────────────────────────────────────────────────────────┤
│  Actor: 历史观测 → Estimator → [obs + vel + latent]     │
│                                       ↓                 │
│                                  MLP → actions          │
│                                                         │
│  (Critic 不需要，Estimator 已学会估计速度)                │
└─────────────────────────────────────────────────────────┘
```

### 二、HIM 网络架构

#### 2.1 HIMEstimator 网络

```python
class HIMEstimator(nn.Module):
    """
    从历史观测中估计速度和隐式状态
    
    输入: obs_history [batch, temporal_steps * num_one_step_obs]
          例如: [batch, 5 * 57] = [batch, 285]
    
    输出: velocity [batch, 3]     - 估计的基座线速度
          latent   [batch, 16]    - 隐式状态表示
    """
    
    def __init__(self):
        # Encoder: 历史观测 → [velocity + latent]
        self.encoder = nn.Sequential(
            nn.Linear(285, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 19)  # 3 (vel) + 16 (latent)
        )
        
        # Target: 当前观测 → latent (用于 SwAV)
        self.target = nn.Sequential(
            nn.Linear(57, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 16)
        )
        
        # Prototype: 可学习的聚类中心
        self.proto = nn.Embedding(16, 16)  # 16 个原型，每个 16 维
```

#### 2.2 HIMActorCritic 网络

```python
class HIMActorCritic(nn.Module):
    """
    HIM 版本的 Actor-Critic 网络
    
    Actor 输入: 当前观测(57D) + 估计速度(3D) + latent(16D) = 76D
    Actor 输出: 动作均值 [batch, num_actions]
    
    Critic 输入: 特权观测（含真实速度）
    Critic 输出: 状态价值 [batch, 1]
    """
    
    def __init__(self):
        # Estimator
        self.estimator = HIMEstimator(...)
        
        # Actor: [current_obs + vel + latent] → actions
        self.actor = nn.Sequential(
            nn.Linear(76, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 16)  # num_actions
        )
        
        # Critic: privileged_obs → value
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # 可学习的动作标准差
        self.std = nn.Parameter(torch.ones(num_actions))
    
    def act(self, obs_history):
        # 1. Estimator 预测速度和 latent
        vel, latent = self.estimator(obs_history)
        
        # 2. 提取当前帧观测
        current_obs = obs_history[:, -57:]
        
        # 3. 拼接输入
        actor_input = torch.cat([current_obs, vel, latent], dim=-1)
        
        # 4. 计算动作
        action_mean = self.actor(actor_input)
        return Normal(action_mean, self.std).sample()
```

### 三、HIM 观测空间详解

#### 3.1 Isaac Lab 输出格式（按变量分组）

```
Isaac Lab 历史观测格式 [batch, 285]:
├── base_ang_vel:     [t-4, t-3, t-2, t-1, t] × 3D = 15D
├── projected_gravity:[t-4, t-3, t-2, t-1, t] × 3D = 15D
├── velocity_commands:[t-4, t-3, t-2, t-1, t] × 3D = 15D
├── joint_pos:        [t-4, t-3, t-2, t-1, t] × 16D = 80D
├── joint_vel:        [t-4, t-3, t-2, t-1, t] × 16D = 80D
└── last_action:      [t-4, t-3, t-2, t-1, t] × 16D = 80D
```

#### 3.2 HIM 需要的格式（按时间步分组）

```
HIM 历史观测格式 [batch, 285]:
├── timestep t-4: [ang_vel, gravity, cmd, jpos, jvel, action] = 57D
├── timestep t-3: [ang_vel, gravity, cmd, jpos, jvel, action] = 57D
├── timestep t-2: [ang_vel, gravity, cmd, jpos, jvel, action] = 57D
├── timestep t-1: [ang_vel, gravity, cmd, jpos, jvel, action] = 57D
└── timestep t-0: [ang_vel, gravity, cmd, jpos, jvel, action] = 57D
```

#### 3.3 观测重排序函数

```python
def reshape_isaac_to_him(obs_flat, history_len, obs_dims):
    """
    将 Isaac Lab 格式转换为 HIM 格式
    
    Args:
        obs_flat: [batch, 285] Isaac Lab 格式
        history_len: 5
        obs_dims: [3, 3, 3, 16, 16, 16]  # 每个变量的维度
    
    Returns:
        obs_him: [batch, 285] HIM 格式
    """
    # 1. 提取每个变量的历史
    var_histories = []
    offset = 0
    for var_dim in obs_dims:  # [3, 3, 3, 16, 16, 16]
        var_flat = obs_flat[:, offset:offset + var_dim * history_len]
        var_history = var_flat.reshape(batch, history_len, var_dim)
        var_histories.append(var_history)
        offset += var_dim * history_len
    
    # 2. 按时间步重组
    timesteps = []
    for t in range(history_len):
        timestep_vars = [var_hist[:, t, :] for var_hist in var_histories]
        timestep_obs = torch.cat(timestep_vars, dim=-1)  # 57D
        timesteps.append(timestep_obs)
    
    obs_him = torch.cat(timesteps, dim=-1)  # 285D
    return obs_him
```

### 四、HIM 训练流程

#### 4.1 数据收集阶段

```python
for step in range(num_steps_per_env):
    # 1. 获取观测
    obs_dict = env.get_observations()
    actor_obs = obs_dict["policy"]    # [batch, 285] 历史观测
    critic_obs = obs_dict["critic"]   # [batch, critic_dim] 特权观测
    
    # 2. 重排序观测 (Isaac Lab → HIM 格式)
    actor_obs_him = reshape_isaac_to_him(actor_obs, history_len=5, obs_dims=[3,3,3,16,16,16])
    
    # 3. 选择动作
    actions = ppo.act(actor_obs_him, critic_obs)
    
    # 4. 环境步进
    next_obs_dict, rewards, dones, infos = env.step(actions)
    
    # 5. 存储转换（包含 next_critic_obs 用于速度监督）
    ppo.process_env_step(rewards, dones, infos, next_critic_obs)
```

#### 4.2 网络更新阶段

```python
def update(self):
    # 遍历 mini-batches
    for batch in storage.mini_batch_generator():
        obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, ... = batch
        
        # ============ 更新 Estimator ============
        # 使用 t+1 时刻的真实速度作为监督信号
        estimation_loss, swap_loss = self.actor_critic.estimator.update(
            obs_history=obs_batch,
            next_critic_obs=next_critic_obs_batch  # 包含 t+1 时刻的速度
        )
        
        # ============ 更新 Actor-Critic (PPO) ============
        # 1. 前向传播
        self.actor_critic.act(obs_batch)
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions_batch)
        value = self.actor_critic.evaluate(critic_obs_batch)
        
        # 2. PPO 损失计算
        ratio = torch.exp(actions_log_prob - old_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        
        value_loss = (value - returns).pow(2).mean()
        
        # 3. 总损失
        loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()
```

### 五、HIM 配置文件说明

#### 5.1 环境配置 (`rough_env_cfg.py`)

```python
@configclass
class MyDogHistObservationsCfg(ObservationsCfg):
    """HIM 版本的观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # 各观测项定义...
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, ...)
        projected_gravity = ObsTerm(...)
        velocity_commands = ObsTerm(...)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_without_wheel, ...)
        joint_vel = ObsTerm(..., scale=0.05)
        actions = ObsTerm(...)
        
        def __post_init__(self):
            self.history_length = 5  # 关键配置！启用 5 帧历史
    
    @configclass
    class CriticCfg(ObsGroup):
        # 必须包含 base_lin_vel 用于监督 Estimator
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, ...)
        # 其他观测项...
        
        def __post_init__(self):
            self.history_length = 5
    
    @configclass
    class HeightScanCfg(ObsGroup):
        # 高度扫描分离到独立组
        height_scan = ObsTerm(...)
```

#### 5.2 PPO 配置 (`rsl_rl_ppo_cfg.py`)

```python
@configclass
class MyDogHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """HIM 版本的 PPO 配置"""
    
    num_steps_per_env = 200  # 与 HIMLoco 论文一致
    max_iterations = 20000
    
    # 观测组分配 - 关键配置！
    obs_groups = {
        "policy": ["policy"],                      # Actor 使用
        "critic": ["critic", "height_scan_group"]  # Critic 使用
    }
    
    algorithm = RslRlPpoAlgorithmCfg(
        max_grad_norm=10.0,  # 与论文一致，不是 1.0
        gamma=0.99,
        learning_rate=1e-3,
        ...
    )
```

### 六、HIM 已注册环境

| 环境 ID | 类型 | 地形 | 训练脚本 |
|---------|------|------|----------|
| `RobotLab-Isaac-Velocity-Flat-MyDog-v0` | 标准 PPO | 平坦 | `train.py` |
| `RobotLab-Isaac-Velocity-Rough-MyDog-v0` | 标准 PPO | 粗糙 | `train.py` |
| `RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0` | **HIM** | 平坦 | `train_him.py` |
| `RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0` | **HIM** | 粗糙 | `train_him.py` |
| `RobotLab-Isaac-Velocity-Handstand-MyDog-v0` | 标准 PPO | 平坦 | `train.py` |

### 七、HIM 训练命令

```bash
# 1. Flat 地形预训练（推荐先用简单环境验证）
python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Flat-MyDog-Hist-v0 \
    --num_envs 4096 \
    --headless

# 2. Rough 地形训练
python scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 \
    --num_envs 4096 \
    --headless

# 3. 可选参数
    --latent_dim 16           # latent 维度
    --estimator_lr 1e-3       # Estimator 学习率
    --num_prototype 16        # SwAV 原型数量
    --estimation_loss_weight 1.0
    --swap_loss_weight 1.0

# 4. 播放/导出策略
python scripts/reinforcement_learning/rsl_rl/play_him.py \
    --task RobotLab-Isaac-Velocity-Rough-MyDog-Hist-v0 \
    --num_envs 64
```

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
  - **支持标准 PPO 和 HIM 框架**
- **当前分支**：`feature/mydog`
- **最近训练**：`logs/rsl_rl/mydog_rough/` 和 `mydog_flat/`

## 构建、测试与开发命令

```bash
# 安装
python -m pip install -e source/robot_lab

# 验证环境注册
python scripts/tools/list_envs.py

# TensorBoard
tensorboard --logdir=logs
```

## 编码风格与命名约定

目标平台为 Linux 上的 Python 3.10+，使用 4 空格缩进并限制 120 列宽。导入使用 isort（含自定义区段）组织，通过 pre-commit 触发 flake8、codespell 与 pyright 基础类型检查。环境命名遵循 `RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v<Version>` 约定，并与 `config/` 下的目录结构保持一致。文件名使用 snake_case；仅为 Isaac/Omniverse API 或基于模式的标识符保留 camelCase。为每个模块添加简洁的文档字符串，描述机器人、任务或智能体配置。

## 测试指南

当新增环境或修改注册表时，将 `list_envs.py` 视为快速冒烟测试。强化学习回归依赖使用固定随机种子的 train 脚本；度量写入 `logs/`，并在训练稳定性相关时分享 tensorboard 日志。使用 `--video` 捕获可视化内容并记录 GPU 数量与硬件声明。

## 提交与 PR 指南

近期提交混合了精简单行与传统 `feat/fix/docs` 前缀。推荐使用语义化提交（例如 `feat: add HIM support`）并提及相关 issue。本地运行 pre-commit，排除生成的检查点或日志，确保新资产存放在正确的品牌目录。PR 需要说明场景、列出验证命令、注明硬件（GPU 数量、Isaac Sim 版本），并附上保存于 `docs/` 的截图或短视频以说明任何可视化变更。

## 安全与配置提示

不要将 API 密钥、个人 Omniverse 路径或专有网格归档提交到仓库。提供下载指引，而不是提交供应商数据。共享检查点时，通过制品存储发布并仅在 PR 中链接；上传前请清理日志中的内部主机名或凭证。
