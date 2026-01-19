# HIM 配置示例

## 基础配置示例

### 示例 1：简单的 Go1 机器人配置

```yaml
# config_him_simple.yaml

policy:
  class_name: "HIMActorCritic"
  
  # 观测格式：时间序列格式
  obs_format:
    policy:
      base_lin_vel: [12]      # 线速度 (x, y, z) * 4 = 12
      base_ang_vel: [3]       # 角速度
      joint_angles: [12]      # 关节角度
      # 总计：27维/步
  
  # 网络配置
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  activation: "elu"
  init_noise_std: 1.0
  
  # HIM 特定配置
  history_size: 10            # 10步历史
  num_one_step_obs: null      # 自动计算：27/10 -> 不整除，需要调整！
```

**问题**：观测维度 27 不能被 history_size 10 整除。

**解决方案 A**：调整观测维度

```yaml
policy:
  obs_format:
    policy:
      obs: [300]  # 30维/步 * 10 = 300维总计
      
  history_size: 10
  num_one_step_obs: 30
```

**解决方案 B**：调整 history_size

```yaml
policy:
  obs_format:
    policy:
      obs: [27]  # 27维/步 * 10 = 270维总计
      
  history_size: 10
  # 27 个维度，但不整除... 选择 9 步
  
# 或者改为
  history_size: 9
  num_one_step_obs: null  # 27 / 9 = 3 ❌ 还是不对
```

最简单的方案：**规范化观测维度**

```yaml
policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      obs: [320]  # 32维/步 * 10 = 320维总计（2的幂，便于计算）
  
  critic:
    # Critic 可以使用特权观测，任意维度
    privileged_obs: [512]
  
  # HIM 参数
  history_size: 10
  num_one_step_obs: 32  # 显式指定或自动计算
  
  # Estimator 网络
  enc_hidden_dims: [128, 64, 16]
  tar_hidden_dims: [128, 64]
  num_prototype: 32
  temperature: 3.0
  
  # Actor 和 Critic 网络
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  activation: "elu"
  init_noise_std: 1.0

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 50
  save_interval: 100
  
  # Normalizer 配置
  normalizers:
    policy:
      class_name: "EmpiricalNormalization"
      shape: [320]

algorithm:
  class_name: "HIMPPO"
  
  # PPO 参数
  num_learning_epochs: 4
  num_mini_batches: 4
  clip_param: 0.2
  gamma: 0.998
  lam: 0.95
  value_loss_coef: 1.0
  entropy_coef: 0.0
  learning_rate: 1e-3
  max_grad_norm: 1.0
```

---

## 高级配置示例

### 示例 2：多奖励 HIM 配置

```yaml
# config_him_multi_reward.yaml

policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      obs: [256]        # 32维/步 * 8 = 256维
    critic:
      priv_obs: [512]
  
  # HIM 参数
  history_size: 8
  num_one_step_obs: 32
  
  # Estimator - 更大的网络
  enc_hidden_dims: [256, 128, 32]   # 更大的隐层
  tar_hidden_dims: [256, 128]
  num_prototype: 64                  # 更多原型
  temperature: 3.5
  
  # 多奖励设置
  num_rewards: 3                     # 3个奖励输出
  
  # 网络大小（较大）
  actor_hidden_dims: [1024, 512, 256]
  critic_hidden_dims: [1024, 512, 256]
  mu_activation: "tanh"              # 限制动作输出范围

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 100
  save_interval: 50
  
  normalizers:
    policy:
      class_name: "EmpiricalNormalization"
      shape: [256]
      epsilon: 1e-8

algorithm:
  class_name: "HIMPPO"
  
  num_learning_epochs: 8
  num_mini_batches: 8
  clip_param: 0.2
  gamma: 0.998
  lam: 0.95
  
  # 多奖励权重
  advantage_mixing_weights: [0.5, 0.3, 0.2]
  
  value_loss_coef: 1.0
  entropy_coef: 0.01
  learning_rate: 5e-4
  max_grad_norm: 1.0
  schedule: "adaptive"
  desired_kl: 0.01
```

---

### 示例 3：轻量级配置（快速迭代）

```yaml
# config_him_lightweight.yaml

policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      obs: [64]         # 只有 64 维观测
    critic:
      obs: [64]
  
  # 最小化 HIM 配置
  history_size: 4       # 更短的历史
  num_one_step_obs: 16
  
  # 较小的 Estimator
  enc_hidden_dims: [64, 32, 8]       # 很小的网络
  tar_hidden_dims: [64, 32]
  num_prototype: 8                    # 最少原型
  temperature: 2.0
  
  # 小网络
  actor_hidden_dims: [256, 128]
  critic_hidden_dims: [256, 128]
  activation: "relu"

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 20              # 较少步数
  save_interval: 200

algorithm:
  class_name: "HIMPPO"
  
  num_learning_epochs: 2             # 较少训练轮次
  num_mini_batches: 2
  learning_rate: 1e-3
```

---

## 环境特定配置

### 示例 4：Quadruped 任务

```yaml
# config_him_quadruped.yaml

env:
  class_name: "VecEnvRobot"
  # 机器人特定参数
  num_envs: 2048

policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      # Quadruped 观测：线速度(3) + 角速度(3) + IMU(9) + 关节角(12) = 27/步
      # 但需要调整为规范格式
      obs: [270]        # 27维/步 * 10 = 270维
    critic:
      privileged_obs: [768]  # 更多特权信息
  
  history_size: 10
  num_one_step_obs: 27
  
  enc_hidden_dims: [128, 64, 16]
  tar_hidden_dims: [128, 64]
  
  actor_hidden_dims: [512, 256]
  critic_hidden_dims: [512, 256]
  
  # 针对四足机器人的动作
  # 12 个关节，每个 PD 控制 = 12 个输出
  num_actions: 12

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 50
  
  normalizers:
    policy:
      class_name: "EmpiricalNormalization"
      shape: [270]

algorithm:
  class_name: "HIMPPO"
  
  num_learning_epochs: 4
  num_mini_batches: 4
  
  # 针对连续控制的参数
  clip_param: 0.2
  gamma: 0.998
  lam: 0.95
  learning_rate: 1e-3
```

---

### 示例 5：双足机器人（Humanoid）

```yaml
# config_him_humanoid.yaml

env:
  class_name: "VecEnvRobot"
  num_envs: 4096

policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      obs: [480]        # 48维/步 * 10 = 480维
    critic:
      proprioceptive: [256]
      external: [256]
  
  history_size: 10
  num_one_step_obs: 48  # 足够的信息用于平衡
  
  # 较大的 Estimator（humanoid 更复杂）
  enc_hidden_dims: [256, 128, 32]
  tar_hidden_dims: [256, 128]
  num_prototype: 64
  temperature: 3.5
  
  # 较大的网络
  actor_hidden_dims: [1024, 512, 256]
  critic_hidden_dims: [1024, 512, 256]
  
  num_actions: 17  # Humanoid 关节数

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 100
  save_interval: 100

algorithm:
  class_name: "HIMPPO"
  
  num_learning_epochs: 5
  num_mini_batches: 8
  learning_rate: 5e-4
```

---

## 对比配置

### 使用标准 ActorCritic vs HIM

```yaml
# 标准 ActorCritic 配置
policy_standard:
  class_name: "ActorCritic"
  obs_format:
    policy:
      obs: [512]        # 任意维度
  actor_hidden_dims: [256, 256]
  critic_hidden_dims: [256, 256]

# HIM 配置（等效复杂度）
policy_him:
  class_name: "HIMActorCritic"
  obs_format:
    policy:
      obs: [320]        # 32维/步 * 10 = 320维
  history_size: 10
  num_one_step_obs: 32
  enc_hidden_dims: [128, 64, 16]    # Estimator 开销
  actor_hidden_dims: [256, 256]     # 较小的 actor
  critic_hidden_dims: [256, 256]
```

**权衡分析**：
- 标准：512维直接输入 → 256维隐层
- HIM：320维输入 → 32维 + 3维速度 + 16维潜在 → 256维隐层

HIM 的 Actor 输入更小（51维 vs 512维），但多了 Estimator 计算。

---

## 调试配置

### 示例 6：最小化配置（验证设置）

```yaml
# config_him_debug.yaml

policy:
  class_name: "HIMActorCritic"
  
  obs_format:
    policy:
      obs: [32]         # 4维/步 * 8 = 32维（最小）
    critic:
      obs: [32]
  
  history_size: 8
  num_one_step_obs: 4
  
  # 最小网络
  enc_hidden_dims: [16, 8, 4]
  tar_hidden_dims: [16, 8]
  num_prototype: 4
  
  actor_hidden_dims: [64, 32]
  critic_hidden_dims: [64, 32]

runner:
  class_name: "OnPolicyRunner"        # 使用标准 runner
  num_steps_per_env: 5               # 最少步数（快速迭代）

algorithm:
  class_name: "HIMPPO"
  
  num_learning_epochs: 1
  num_mini_batches: 1
  learning_rate: 1e-3
```

用于快速验证配置是否有效。

---

## 配置检查清单

创建 HIM 配置时，确保：

```
☐ obs_format.policy 维度 % history_size == 0
  示例：obs_size / history_size = num_one_step_obs
  
☐ history_size 合理（通常 5-20）

☐ num_one_step_obs 与 estimator 输入匹配

☐ 如果使用 Critic 观测，维度足够大

☐ enc_hidden_dims[-1] 合理（通常 8-64）

☐ num_prototype 足够（通常 16-64）

☐ 网络大小与可用内存匹配

☐ 学习率适当（通常 1e-4 ~ 1e-3）

☐ 如果使用多 GPU，batch_size 足够大
```

---

## 快速参考

### 参数推荐表

| 任务规模 | history_size | num_one_step_obs | enc_hidden_dims | num_prototype |
|---------|-------------|------------------|-----------------|---------------|
| 小型 | 4-6 | 8-16 | [64, 32, 8] | 8-16 |
| 中型 | 8-10 | 16-32 | [128, 64, 16] | 32 |
| 大型 | 10-15 | 32-64 | [256, 128, 32] | 64 |

### 内存估算（单位：MB）

```
Estimator 参数：
  ≈ (temporal_steps * num_one_step_obs) * enc_hidden_dims[0]
  + enc_hidden_dims 中间计算
  ≈ 1-10 MB （取决于大小）

Actor 参数：
  ≈ (num_one_step_obs + 3 + enc_hidden_dims[-1]) * actor_hidden_dims[0]
  ≈ 100-500 KB

总开销：通常 < 50 MB
```

---

**配置版本**：1.0
**最后更新**：2026-01-19
