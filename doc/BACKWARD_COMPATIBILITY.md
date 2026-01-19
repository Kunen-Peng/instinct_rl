# 向后兼容性验证指南

## 确保 HIM 模块与现有代码的兼容性

### 1. PPO 算法兼容性

#### 标准 PPO 使用（现有代码）
```python
from instinct_rl.algorithms import PPO

alg = PPO(actor_critic, device="cuda")
alg.process_env_step(rewards, dones, infos, obs, critic_obs)
```

#### HIMPPO 使用（新代码）
```python
from instinct_rl.algorithms import HIMPPO

alg = HIMPPO(actor_critic, device="cuda")
# 不提供 next_critic_obs_for_bootstrap 时行为与 PPO 相同
alg.process_env_step(rewards, dones, infos, obs, critic_obs)

# 提供 next_critic_obs_for_bootstrap 时使用 HIM 功能
alg.process_env_step(rewards, dones, infos, obs, critic_obs, next_critic_obs_for_bootstrap)
```

**向后兼容性保证**：
- ✅ HIMPPO 的 `process_env_step` 添加了可选参数 `next_critic_obs_for_bootstrap`
- ✅ 不提供该参数时行为与标准 PPO 相同
- ✅ 可以直接替换现有的 PPO

### 2. Runner 兼容性

#### 标准 OnPolicyRunner 使用（现有代码）
```python
from instinct_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(env, train_cfg, log_dir="logs", device="cuda")
runner.learn(num_learning_iterations=1000)
```

#### HIMOnPolicyRunner 使用（新代码）
```python
from instinct_rl.runners import HIMOnPolicyRunner

runner = HIMOnPolicyRunner(env, train_cfg, log_dir="logs", device="cuda")
runner.learn(num_learning_iterations=1000)
```

**向后兼容性保证**：
- ✅ HIMOnPolicyRunner 继承所有 OnPolicyRunner 的功能
- ✅ 支持所有相同的初始化参数
- ✅ 支持相同的 API（learn, save, load 等）
- ✅ `use_termination_obs` 默认为 False，禁用 HIM 特性时行为与标准 runner 相同

### 3. 配置兼容性

#### 现有配置（标准 PPO）
```yaml
runner:
  class_name: "OnPolicyRunner"
  num_steps_per_env: 24
  save_interval: 100
  
algorithm:
  class_name: "PPO"
  gamma: 0.998
  lam: 0.95
```

#### 新配置（HIMPPO）
```yaml
runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 24
  save_interval: 100
  use_termination_obs: false  # 禁用时与现有配置兼容
  
algorithm:
  class_name: "HIMPPO"
  gamma: 0.998
  lam: 0.95
```

**向后兼容性保证**：
- ✅ 可以在现有代码中用 HIMPPO 替换 PPO
- ✅ 可以用 HIMOnPolicyRunner 替换 OnPolicyRunner
- ✅ 不需要修改其他配置参数

### 4. 检查点兼容性

```python
# 用 PPO 保存的检查点可以用 HIMPPO 加载
ppo_runner = OnPolicyRunner(env, cfg, device="cuda")
ppo_runner.load("ppo_checkpoint.pt")

# 转换为 HIMPPO 继续训练
him_runner = HIMOnPolicyRunner(env, cfg, device="cuda")
him_runner.load("ppo_checkpoint.pt")
him_runner.learn(100)

# 反之亦然
him_runner.save("him_checkpoint.pt")
ppo_runner.load("him_checkpoint.pt")  # 在不使用 HIM 特性时可正常加载
```

**向后兼容性保证**：
- ✅ HIMPPO 与 PPO 的 state_dict 兼容
- ✅ 可以在两种算法间自由迁移检查点

### 5. 验证步骤

#### 测试 1：确认 PPO 仍可正常使用
```python
from instinct_rl.algorithms import PPO

alg = PPO(actor_critic, device="cuda")
alg.init_storage(num_envs=64, num_transitions_per_env=24, ...)
alg.process_env_step(rewards, dones, infos, obs, critic_obs)
# ✓ 应能正常运行
```

#### 测试 2：确认 HIMPPO 可作为 PPO 的替代品
```python
from instinct_rl.algorithms import HIMPPO

alg = HIMPPO(actor_critic, device="cuda")
alg.init_storage(num_envs=64, num_transitions_per_env=24, ...)
alg.process_env_step(rewards, dones, infos, obs, critic_obs)
# ✓ 应能正常运行，行为与 PPO 相同
```

#### 测试 3：确认 OnPolicyRunner 仍可正常使用
```python
from instinct_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(env, train_cfg, device="cuda")
runner.learn(num_learning_iterations=10)
# ✓ 应能正常运行
```

#### 测试 4：确认 HIMOnPolicyRunner 可作为 OnPolicyRunner 的替代品
```python
from instinct_rl.runners import HIMOnPolicyRunner

runner = HIMOnPolicyRunner(env, train_cfg, device="cuda")
runner.learn(num_learning_iterations=10)
# ✓ 应能正常运行，行为与 OnPolicyRunner 相同
```

### 6. 迁移指南

#### 从 PPO 升级到 HIMPPO（无 termination obs）

最小改动：仅改变 class_name

```yaml
# 之前
algorithm:
  class_name: "PPO"

# 之后
algorithm:
  class_name: "HIMPPO"
```

#### 从 OnPolicyRunner 升级到 HIMOnPolicyRunner（无 termination obs）

最小改动：仅改变 class_name

```yaml
# 之前
runner:
  class_name: "OnPolicyRunner"

# 之后
runner:
  class_name: "HIMOnPolicyRunner"
```

#### 启用 Termination Observations

1. 在环境配置中添加 `termination_observations`
2. 在 runner 配置中设置 `use_termination_obs: true`

```yaml
runner:
  class_name: "HIMOnPolicyRunner"
  use_termination_obs: true
```

### 7. 监控兼容性

在运行实验时检查以下指标：

```python
# 标准 PPO
ppo_rewards = [...]
ppo_loss = [...]

# HIMPPO（不使用 termination obs）
him_no_term_rewards = [...]
him_no_term_loss = [...]

# 验证：him_no_term_rewards ≈ ppo_rewards（在统计上）
# 如果相近，说明完全兼容
```

## 总结

- **完全向后兼容**：现有代码可无缝升级到 HIM
- **可选功能**：termination observations 完全可选
- **渐进升级**：可以先不启用 HIM 特性，后续逐步启用
- **检查点兼容**：PPO 和 HIMPPO 检查点可相互兼容

这种设计确保了在不影响现有代码的情况下，为 Instinct-RL 添加了强大的新功能。
