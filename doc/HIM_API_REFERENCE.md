# HIM 模块 API 参考

## HIMEstimator 类

### 概述

从观测历史中提取速度和潜在特征的神经网络模块。使用对比学习通过原型匹配来学习表示。

### 导入

```python
from instinct_rl.modules import HIMEstimator
```

### 初始化

```python
estimator = HIMEstimator(
    temporal_steps: int,           # 观测历史步数
    num_one_step_obs: int,         # 单步观测维度
    enc_hidden_dims: list = [128, 64, 16],     # 编码器隐层
    tar_hidden_dims: list = [128, 64],         # 目标网络隐层
    activation: str = "elu",       # 激活函数
    learning_rate: float = 1e-3,   # 优化器学习率
    max_grad_norm: float = 10.0,   # 梯度裁剪范数
    num_prototype: int = 32,       # 原型数量
    temperature: float = 3.0,      # softmax 温度
)
```

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| temporal_steps | int | 观测历史长度 | 必需 |
| num_one_step_obs | int | 单步观测维度 | 必需 |
| enc_hidden_dims | list | 编码器网络层维度 | [128, 64, 16] |
| tar_hidden_dims | list | 目标网络层维度 | [128, 64] |
| activation | str | 激活函数（elu/relu/tanh等） | "elu" |
| learning_rate | float | Adam 优化器学习率 | 1e-3 |
| max_grad_norm | float | 梯度裁剪的最大范数 | 10.0 |
| num_prototype | int | 对比学习中的原型数量 | 32 |
| temperature | float | 原型匹配的温度参数 | 3.0 |

### 方法

#### forward(obs_history)

**用途**：推理模式下处理观测历史

**参数**：
- `obs_history` (Tensor): 形状为 [batch_size, temporal_steps * num_one_step_obs]

**返回**：
- `vel` (Tensor): 估计的速度，形状 [batch_size, 3]
- `z` (Tensor): 标准化的潜在特征，形状 [batch_size, enc_hidden_dims[-1]]

**说明**：返回的张量已分离（detached），适合推理

```python
vel, latent = estimator(obs_history)  # 推理使用
```

#### get_latent(obs_history)

**用途**：获取速度和潜在特征（推理用）

**参数**：
- `obs_history` (Tensor): 观测历史

**返回**：
- `vel` (Tensor): 分离的速度张量
- `z` (Tensor): 分离的潜在张量

**说明**：等同于 forward，专为推理设计

```python
vel, latent = estimator.get_latent(obs_history)
```

#### encode(obs_history)

**用途**：编码观测历史（训练用）

**参数**：
- `obs_history` (Tensor): 观测历史

**返回**：
- `vel` (Tensor): 速度估计（未分离）
- `z` (Tensor): 潜在特征（未分离，用于反向传播）

**说明**：返回的张量未分离，用于训练时反向传播

```python
vel, latent = estimator.encode(obs_history)
# 可以对 latent 计算梯度
loss = some_loss_fn(latent)
loss.backward()
```

#### update(obs_history, next_critic_obs, lr=None)

**用途**：使用对比学习更新 estimator

**参数**：
- `obs_history` (Tensor): 观测历史，形状 [batch, temporal_steps * num_one_step_obs]
- `next_critic_obs` (Tensor): 下一步 critic 观测，形状 [batch, critic_obs_dim]
  - 预期格式：[current_obs_features (3:num_one_step_obs+3), velocity (num_one_step_obs:num_one_step_obs+3), ...]
- `lr` (float, optional): 临时学习率（覆盖当前学习率）

**返回**：
- `estimation_loss` (float): MSE 损失（速度估计）
- `swap_loss` (float): 对比损失

**说明**：执行一步梯度更新

```python
est_loss, cont_loss = estimator.update(obs_history, next_critic_obs)
print(f"Estimation loss: {est_loss}, Contrastive loss: {cont_loss}")
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| temporal_steps | int | 观测历史长度 |
| num_one_step_obs | int | 单步观测维度 |
| num_latent | int | 潜在特征维度 = enc_hidden_dims[-1] |
| encoder | nn.Module | 编码器网络 |
| target | nn.Module | 目标网络 |
| proto | nn.Embedding | 原型嵌入层 |
| optimizer | optim.Optimizer | Adam 优化器 |

### 使用示例

```python
import torch
from instinct_rl.modules import HIMEstimator

# 创建 estimator
estimator = HIMEstimator(
    temporal_steps=10,
    num_one_step_obs=15,
    enc_hidden_dims=[128, 64, 16],
)

# 推理
obs_history = torch.randn(32, 150)  # batch_size=32, 10*15=150
vel, latent = estimator(obs_history)
print(vel.shape)      # [32, 3]
print(latent.shape)   # [32, 16]

# 训练
next_critic_obs = torch.randn(32, 512)  # 包含特权信息
est_loss, cont_loss = estimator.update(obs_history, next_critic_obs, lr=1e-4)
print(f"Loss: {est_loss + cont_loss}")
```

---

## HIMActorCritic 类

### 概述

继承自 ActorCritic 的网络架构，集成 HIMEstimator 用于观测处理。Policy 输入为 [当前观测, 速度, 潜在特征]。

### 导入

```python
from instinct_rl.modules import HIMActorCritic
```

### 初始化

```python
actor_critic = HIMActorCritic(
    obs_format: dict,                    # 观测格式
    num_actions: int,                    # 动作维度
    actor_hidden_dims: list = [512, 256, 128],
    critic_hidden_dims: list = [512, 256, 128],
    activation: str = "elu",
    init_noise_std: float = 1.0,
    num_rewards: int = 1,
    # HIM 特定参数
    history_size: int = 10,              # 观测历史步数
    num_one_step_obs: int = None,        # 单步维度（自动计算）
    enc_hidden_dims: list = [128, 64, 16],
    tar_hidden_dims: list = [128, 64],
    num_prototype: int = 32,
    temperature: float = 3.0,
)
```

### 参数说明

#### 基础参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| obs_format | dict | 观测格式字典 | 必需 |
| num_actions | int | 动作维度 | 必需 |
| actor_hidden_dims | list | Actor 网络隐层 | [512, 256, 128] |
| critic_hidden_dims | list | Critic 网络隐层 | [512, 256, 128] |
| activation | str | 激活函数 | "elu" |
| init_noise_std | float | 初始动作标准差 | 1.0 |
| num_rewards | int | 奖励个数（支持多奖励） | 1 |

#### HIM 特定参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| history_size | int | 观测历史长度 | 10 |
| num_one_step_obs | int | 单步观测维度（None 自动计算） | None |
| enc_hidden_dims | list | Estimator 编码器隐层 | [128, 64, 16] |
| tar_hidden_dims | list | Estimator 目标网络隐层 | [128, 64] |
| num_prototype | int | 对比学习原型数 | 32 |
| temperature | float | softmax 温度 | 3.0 |

### 方法

#### act(obs_history, **kwargs)

**用途**：采样动作（训练）

**参数**：
- `obs_history` (Tensor): 观测历史 [batch, history_size * num_one_step_obs]
- `**kwargs`: 其他参数（保留接口）

**返回**：
- 采样的动作 (Tensor): [batch, num_actions]

```python
actions = actor_critic.act(obs_history)
```

#### act_inference(obs_history, **kwargs)

**用途**：确定性动作选择（推理）

**参数**：
- `obs_history` (Tensor): 观测历史

**返回**：
- 平均动作 (Tensor): [batch, num_actions]

**说明**：返回分布的均值，不采样，用于推理

```python
actions_mean = actor_critic.act_inference(obs_history)
```

#### evaluate(critic_observations, **kwargs)

**用途**：评估值函数

**参数**：
- `critic_observations` (Tensor 或 list): Critic 观测
  - 单奖励：Tensor [batch, critic_obs_dim]
  - 多奖励：Tensor [batch, critic_obs_dim] 或 list of Tensors

**返回**：
- 值估计 (Tensor): [batch, num_rewards]

```python
values = actor_critic.evaluate(critic_obs)
```

#### get_actions_log_prob(actions)

**用途**：获取动作的对数概率

**参数**：
- `actions` (Tensor): 动作 [batch, num_actions]

**返回**：
- 对数概率 (Tensor): [batch]

```python
log_probs = actor_critic.get_actions_log_prob(actions)
```

#### update_distribution(obs_history)

**用途**：更新动作分布（内部方法）

**参数**：
- `obs_history` (Tensor): 观测历史

**说明**：由 act() 自动调用，通常不需要显式调用

#### reset(dones=None)

**用途**：重置内部状态

**参数**：
- `dones` (Tensor, optional): done 标志

**说明**：为了兼容性设计（recurrent 网络用），当前不做任何操作

#### clip_std(min=None, max=None)

**用途**：裁剪动作标准差

**参数**：
- `min` (float, optional): 最小值
- `max` (float, optional): 最大值

```python
actor_critic.clip_std(min=0.01, max=1.0)
```

#### export_as_onnx(observations, filedir)

**用途**：导出为 ONNX 格式

**参数**：
- `observations` (Tensor): 示例观测
- `filedir` (str): 导出目录

**说明**：导出 estimator 和 actor 网络

```python
actor_critic.export_as_onnx(obs_sample, "./onnx_models")
# 生成：him_estimator.onnx, actor.onnx
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| estimator | HIMEstimator | HIM estimator 模块 |
| actor | nn.Module | Policy 网络 |
| critic / critics | nn.Module / list | 值网络 |
| std | nn.Parameter | 动作标准差 |
| history_size | int | 观测历史长度 |
| num_one_step_obs | int | 单步观测维度 |
| action_mean | Tensor | 当前分布的均值 |
| action_std | Tensor | 当前分布的标准差 |
| entropy | Tensor | 当前分布的熵 |
| obs_segments | dict | Policy 观测段 |
| critic_obs_segments | dict | Critic 观测段 |

### 使用示例

```python
import torch
from instinct_rl.modules import HIMActorCritic

# 定义观测格式
obs_format = {
    "policy": {"obs": (150,)},           # 10 步 * 15 维
    "critic": {"priv_obs": (512,)}       # 特权观测
}

# 创建网络
actor_critic = HIMActorCritic(
    obs_format=obs_format,
    num_actions=12,
    history_size=10,
    num_one_step_obs=15,
)

# 推理
obs_history = torch.randn(32, 150)
actions = actor_critic.act(obs_history)

# 值评估
critic_obs = torch.randn(32, 512)
values = actor_critic.evaluate(critic_obs)

# 确定性推理
actions_mean = actor_critic.act_inference(obs_history)
```

---

## 框架集成

### 通过 modules.build_actor_critic 创建

```python
from instinct_rl import modules

actor_critic = modules.build_actor_critic(
    policy_class_name="HIMActorCritic",
    policy_cfg={
        "history_size": 10,
        "num_one_step_obs": 15,
        "actor_hidden_dims": [512, 256],
        "critic_hidden_dims": [512, 256],
    },
    obs_format=obs_format,
    num_actions=12,
    num_rewards=1,
)
```

### 与 PPO 算法集成

```python
from instinct_rl.algorithms import PPO

alg = PPO(actor_critic, device="cuda")

# 标准 PPO 接口
actions = alg.act(obs, critic_obs)
alg.process_env_step(rewards, dones, infos, obs, critic_obs)
values = alg.actor_critic.evaluate(critic_obs)
```

### 与 Runner 集成

```python
from instinct_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(env, train_cfg)
runner.learn(num_learning_iterations=1000)

# 自动处理 HIMActorCritic 的前向传播
```

---

## 常见用法模式

### 模式 1：基础训练

```python
# 1. 创建网络
actor_critic = HIMActorCritic(obs_format, num_actions, history_size=10)

# 2. 创建算法和 runner
alg = PPO(actor_critic)
runner = OnPolicyRunner(env, config)

# 3. 训练
runner.learn(num_iterations)
```

### 模式 2：自定义 Estimator 学习率

```python
# 在 runner 中添加 estimator 更新
est_lr = 5e-4
for _ in range(num_updates):
    actor_critic.estimator.update(
        obs_history, 
        next_critic_obs,
        lr=est_lr
    )
```

### 模式 3：模型保存和加载

```python
# 保存
torch.save(actor_critic.state_dict(), "him_model.pt")

# 加载
actor_critic.load_state_dict(torch.load("him_model.pt"))

# 推理
actor_critic.eval()
with torch.no_grad():
    actions = actor_critic.act_inference(obs_history)
```

---

## 性能调优

### 关键超参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| history_size | 5-20 | 更大 = 更多上下文，但计算成本增加 |
| num_one_step_obs | 8-32 | 应该是 policy_obs_size / history_size |
| enc_hidden_dims | [64-256] | 更大 = 更好的特征，但更多参数 |
| num_prototype | 16-64 | 更多原型 = 更丰富的表示 |
| temperature | 2.0-5.0 | 更低 = 更尖锐的分布 |
| estimator_lr | 1e-4 ~ 1e-3 | 通常小于策略学习率 |

### 内存优化

```python
# 减少 estimator 复杂度
actor_critic = HIMActorCritic(
    obs_format,
    num_actions,
    enc_hidden_dims=[64, 32, 16],    # 更小
    tar_hidden_dims=[64, 32],         # 更小
    num_prototype=16,                 # 更少
)
```

---

**最后更新**：2026-01-19
**API 版本**：1.0
