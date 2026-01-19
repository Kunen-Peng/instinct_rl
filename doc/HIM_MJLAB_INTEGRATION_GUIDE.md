# 使用优化后的 HIMPPO 与 mjlab ObservationManager 集成指南

## 📌 概述

本指南说明如何在 mjlab 环境中使用优化后的 HIMPPO，充分利用 ObservationManager 的观测历史管理。

## 🔧 环境设置

### 第一步：理解 ObservationManager 的观测历史

mjlab 的 ObservationManager 管理观测历史，格式如下：

```python
# 在 mjlab 中配置观测
observations:
  policy:
    concatenate_terms: true
    flatten_history_dim: true          # ← 关键：展平为 1D
    history_length: 10                 # ← 关键：存储 10 步历史
    terms:
      state:
        func: compute_observations    # 计算当前状态
        
  critic:
    concatenate_terms: true
    terms:
      state:
        func: compute_observations
```

### 第二步：验证 CircularBuffer 输出

ObservationManager 使用 CircularBuffer，输出格式为 **oldest_first**：

```
CircularBuffer.buffer
[batch_size, history_length, obs_dim]
        ↓
展平（flatten_history_dim=true）
        ↓
[batch_size, history_length * obs_dim]
[obs_t0, obs_t1, ..., obs_t9]  ← oldest_first 顺序
```

**验证方法**：
```python
# 在 mjlab 环境中检查
obs = env.get_observations()  # 返回 policy obs
obs_shape = obs["policy"].shape
# 应该是 [num_envs, history_length * num_one_step_obs]
```

## 🎯 配置 Instinct-RL

### 第一步：在 Instinct-RL 中设置 HIMActorCritic

```yaml
# train_config.yaml
policy:
  class_name: "HIMActorCritic"
  
  # 必须与 mjlab 的 history_length 一致！
  history_size: 10
  
  # 自动计算或显式指定
  # num_one_step_obs = total_obs_dim / history_size
  num_one_step_obs: 32  # 如果总维度是 320，则 320/10=32
  
  # Actor 和 Critic 隐藏层维度
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  
  # HIMEstimator 配置
  enc_hidden_dims: [128, 64, 16]      # Encoder 隐藏层
  tar_hidden_dims: [128, 64]          # Target encoder 隐藏层
  num_prototype: 32                   # 原型数量（对比学习）
  temperature: 3.0                    # 温度参数
  
  # 一般不需要改
  activation: "elu"
  init_noise_std: 1.0
```

### 第二步：配置 HIMPPO

```yaml
algorithm:
  class_name: "HIMPPO"
  
  # PPO 超参数
  lr: 1e-4
  gamma: 0.99
  lam: 0.95
  entropy_coef: 0.0
  
  # 其他 PPO 参数保持不变
  # ...
```

### 第三步：配置 HIMOnPolicyRunner（可选）

```yaml
runner:
  class_name: "HIMOnPolicyRunner"  # 或使用标准 OnPolicyRunner
  
  # 是否使用 termination obs 进行更准确的 bootstrap
  use_termination_obs: true
  
  # 其他 runner 参数
  num_steps_per_env: 24
  save_interval: 100
  log_interval: 10
```

## 🔗 数据流整合

### mjlab → Instinct-RL 的观测流

```
mjlab 环境
    ↓
ObservationManager.compute()
    ↓
CircularBuffer (每一步自动更新)
    ↓
flattened 历史 [obs_t0, obs_t1, ..., obs_t9]  (oldest_first)
    ↓
env.get_observations() 返回给 Instinct-RL
    ↓
HIMOnPolicyRunner.rollout_step()
    ↓
HIMActorCritic
    ├─ HIMEstimator._prepare_obs_input()
    │  └─ 确认格式是 oldest_first
    ├─ 提取最新观测: obs[:, -num_one_step_obs:]
    ├─ HIMEstimator.forward()
    │  └─ 输出 [vel(3), latent(D)]
    ├─ 拼接: [current_obs, vel, latent]
    └─ Actor/Critic 网络
```

## 📝 完整配置示例

### mjlab 端配置（`scene_config.yaml` 或环境配置）

```yaml
observations:
  policy:
    concatenate_terms: true
    concatenate_dim: -1
    flatten_history_dim: true          # ✓ 必须是 true
    history_length: 10                 # ✓ 存储 10 步
    
    terms:
      # 假设总观测维度 = 320，则单步维度 = 320/10 = 32
      position:
        func: get_robot_position       # 维度 12
      velocity:
        func: get_robot_velocity       # 维度 12
      contact:
        func: get_contact_state        # 维度 8
      # 总共 32 维
  
  critic:
    concatenate_terms: true
    terms:
      position:
        func: get_robot_position
      velocity:
        func: get_robot_velocity
      contact:
        func: get_contact_state
```

### Instinct-RL 端配置（`train_config.yaml`）

```yaml
policy:
  class_name: "HIMActorCritic"
  
  # 关键：必须与 mjlab 的 history_length 一致
  history_size: 10
  
  # 自动计算：320 / 10 = 32
  # 或显式设置
  num_one_step_obs: 32
  
  # 其他参数...
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  enc_hidden_dims: [128, 64, 16]
  tar_hidden_dims: [128, 64]

algorithm:
  class_name: "HIMPPO"
  # PPO 参数...

runner:
  class_name: "HIMOnPolicyRunner"
  use_termination_obs: true
```

## ✅ 验证清单

使用前确保：

- [ ] **ObservationManager 配置**
  - [ ] `flatten_history_dim: true`
  - [ ] `history_length: 10` (或其他值)
  - [ ] `concatenate_terms: true` (if combining multiple obs terms)

- [ ] **HIMActorCritic 配置**
  - [ ] `history_size` = ObservationManager 的 `history_length`
  - [ ] `num_one_step_obs` 正确（= 总观测维度 / history_size）
  - [ ] `enc_hidden_dims[-1]` 与其他参数兼容

- [ ] **观测维度验证**
  ```python
  policy_obs_dim = env.get_obs_format()["policy"]["state"][0]
  num_one_step_obs = policy_obs_dim // history_size
  # 应该能整除，无余数
  assert policy_obs_dim == num_one_step_obs * history_size
  ```

- [ ] **运行验证脚本**
  ```bash
  python test_observation_ordering.py
  # 所有 5 个测试应该通过
  ```

- [ ] **检查初始化日志**
  - 不应该出现"WARNING"关于观测维度
  - 应该看到"Auto-computed num_one_step_obs"（如果自动计算）

## 🚀 训练

### 使用优化后的 HIMPPO 开始训练

```bash
# 基本用法
python -m space_mjlab.scripts.instinct_rl.train \
  Mjlab-Velocity-Rough-Unitree-Go2-InstinctRL \
  --env.scene.num-envs 2048

# 或带有自定义配置
python -m space_mjlab.scripts.instinct_rl.train \
  Mjlab-Velocity-Rough-Unitree-Go2-InstinctRL \
  --config custom_train_config.yaml \
  --env.scene.num-envs 2048

# 恢复训练
python -m space_mjlab.scripts.instinct_rl.train \
  Mjlab-Velocity-Rough-Unitree-Go2-InstinctRL \
  --checkpoint path/to/checkpoint.pt
```

### 监控训练

```bash
# 在另一个终端查看 TensorBoard
tensorboard --logdir logs/instinct_rl/
```

## 🔍 故障排查

### 问题 1：观测维度不匹配

**症状**：
```
[HIMActorCritic WARNING] Policy obs size (320) != history_size (10) * num_one_step_obs (30)
```

**原因**：
- `num_one_step_obs` 计算错误
- ObservationManager 返回的维度与预期不同

**解决方案**：
```python
# 正确计算
actual_obs_dim = env.get_obs_format()["policy"]["state"][0]
num_one_step_obs = actual_obs_dim // history_size
print(f"Correct num_one_step_obs: {num_one_step_obs}")
```

### 问题 2：观测顺序错误

**症状**：
```
❌ Observation ordering is incorrect
❌ current_obs extraction failed
```

**原因**：
- ObservationManager 的 `flatten_history_dim` 不是 `true`
- 观测顺序不是 oldest_first

**解决方案**：
确保 mjlab 配置：
```yaml
observations:
  policy:
    flatten_history_dim: true  # ✓ 必须是 true
```

### 问题 3：训练不收敛

**症状**：
- 奖励无法增长
- 损失值异常

**原因**：
- 特征维度不匹配
- ObservationManager 和 Instinct-RL 的 history_size 不一致
- 观测顺序不正确

**解决方案**：
```bash
# 1. 运行验证脚本
python test_observation_ordering.py

# 2. 检查日志中的维度信息
# 查找 "HIMActorCritic" 和 "HIMEstimator" 的输出

# 3. 确认所有历史长度相同
grep "history_size" logs/instinct_rl/*.log
```

## 📊 性能优化建议

### 1. Estimator 学习率

如果 estimator 学习不好，调整：

```yaml
policy:
  enc_hidden_dims: [128, 64, 16]  # 增加容量
  num_prototype: 64               # 增加原型数（通常更好）
  temperature: 2.0                # 降低温度（更尖锐）
```

### 2. Actor-Critic 大小

根据任务复杂度调整：

```yaml
# 简单任务
actor_hidden_dims: [256, 128]
critic_hidden_dims: [256, 128]

# 复杂任务
actor_hidden_dims: [1024, 512, 256]
critic_hidden_dims: [1024, 512, 256]
```

### 3. 历史长度

- **短历史 (5-10)**：快速反应，低延迟
- **长历史 (15-30)**：更好的动作预测

```yaml
# mjlab
history_length: 15

# instinct_rl
history_size: 15
num_one_step_obs: 32  # 320 / 15 ≈ 21 (需要调整观测)
```

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| HIM_OPTIMIZATION_SUMMARY.md | 优化总结 |
| HIM_OPTIMIZATION_WITH_OBSERVATION_MANAGER.md | 详细优化说明 |
| HIM_OPTIMIZATION_QUICK_REFERENCE.md | 快速参考 |
| HIM_QUICKSTART.md | 快速开始 |
| HIM_API_REFERENCE.md | API 参考 |
| HIM_CONFIG_EXAMPLES.md | 配置示例 |

## 💡 最佳实践

1. **总是运行验证脚本**
   ```bash
   python test_observation_ordering.py
   ```

2. **检查初始化日志**
   第一次运行时查看是否有维度警告

3. **一致的历史长度**
   确保 mjlab 和 Instinct-RL 使用相同的 history_length

4. **备份配置**
   保存有效的配置文件以便后续参考

5. **逐步调试**
   如果不收敛，逐个改变超参数而不是同时改多个

## 🎯 常见工作流

### 从标准 ActorCritic 迁移到 HIM

```yaml
# 旧配置
policy:
  class_name: "ActorCritic"

# 新配置（最小改动）
policy:
  class_name: "HIMActorCritic"
  history_size: 10           # 新增
  num_one_step_obs: 32       # 新增
  # 其他参数保持不变
```

### 配置 mjlab 支持 HIM

```yaml
# 只需两个改动
observations:
  policy:
    history_length: 10           # ← 新增或改变值
    flatten_history_dim: true    # ← 确保是 true
    
    # 其他配置保持不变
    terms:
      state:
        func: compute_observations
```

---

**文档版本**: 1.0  
**最后更新**: 2026-01-19  
**状态**: ✅ 完成和验证

祝您训练顺利！如有问题，请参考相关文档或运行验证脚本。
