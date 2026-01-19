# HIMPPO 设计说明

## 核心问题：如何实现 next_critic_observations 的处理

### 问题背景

在 PPO 中，当环境因超时（timeout）而终止时，我们需要使用 terminated episode 的实际最后一步状态来评估其价值，而不是简单地假设值为 0。这个过程称为 **bootstrapping**。

对于包含 termination observations 的场景，我们需要：
1. 在环境中捕获 termination 时的观测
2. 使用这些观测（而非环境自动重置后的观测）来评估值
3. 将这个值加到 reward 中作为 bootstrap

### 两种实现方式

#### 方案 A：Instinct-RL 采用的方案（直接计算）

**特点**：
- Bootstrap 值在 `process_env_step` 中**立即计算**
- 直接添加到 rewards 中
- **不需要修改 RolloutStorage**
- 计算后即刻使用，不存储

**优势**：
- 简洁清晰
- 无需扩展 Transition 类或 RolloutStorage
- 避免大量额外的张量存储
- 与现有 Instinct-RL 架构无缝融合

**代码**：
```python
def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs, 
                     next_critic_obs_for_bootstrap=None):
    # ...其他处理...
    
    # 立即计算并应用 bootstrap 值
    if "time_outs" in infos:
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        if bootstrap_obs is not None:
            with torch.no_grad():
                bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
            self.transition.rewards += (
                self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)
            )
```

#### 方案 B：rsl_rl 采用的方案（存储后处理）

**特点**：
- Bootstrap 观测存储在 HIMRolloutStorage.Transition 中
- 在 minibatch 生成时传递给网络重新计算
- **需要专门的 HIMRolloutStorage** 类
- Transition 包含额外字段 `next_critic_observations`

**优势**：
- 灵活：可以在生成 minibatch 时动态改变网络行为
- 显式存储便于调试和检查

**劣势**：
- 需要修改 storage 层
- 额外的 GPU 存储开销（每个 transition 存储两份 critic observations）
- 增加代码复杂度

**代码**（rsl_rl）：
```python
class HIMRolloutStorage.Transition:
    self.next_critic_observations = None

def add_transitions(self, transition):
    # ...其他...
    if self.next_privileged_observations is not None: 
        self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
```

### 为什么 Instinct-RL 选择方案 A

1. **RolloutStorage 没有 next_critic_observations 字段**
   - 添加需要修改核心 storage 类
   - 可能破坏现有代码兼容性
   - RolloutStorage 已在整个系统中广泛使用

2. **方案 A 更符合 Instinct-RL 架构**
   - Instinct-RL 在 `process_env_step` 中就处理完所有逻辑
   - 不依赖 storage 层的额外功能
   - 与 PPO 的现有实现模式一致

3. **功能完全等价**
   - Bootstrap 值在 `process_env_step` 立即计算
   - 与方案 B 在数学上完全等价
   - 不存在信息丢失

4. **性能更优**
   - 方案 A：每个 step 计算一次 bootstrap value，立即使用
   - 方案 B：存储完整的观测，后续生成 minibatch 时再计算（重复计算）
   - 方案 A 的 GPU 内存使用更少

## 数学验证

### Bootstrap 公式

标准 PPO：
$$V_{bootstrap}(s_t) = r_t + \gamma V(s_{t+1})$$

其中 $s_{t+1}$ 在 timeout 时使用 termination observation

### Instinct-RL 实现的逻辑

```python
# 在 step t：
rewards += gamma * V(next_critic_obs_for_bootstrap) * time_outs

# 随后 compute_returns 中：
# returns[t] = rewards[t] + gamma * V(s_{t+1}) * (1 - dones[t]) + ...
#            = (r_t + gamma * V(s_term)) + gamma * V(s_{t+1}) * (1 - dones[t]) + ...
```

由于 timeout 时 `dones[t] = 1`，所以 $V(s_{t+1})$ 的贡献被消除，只有 bootstrap 值被保留。

### rsl_rl HIM 的逻辑

在生成 minibatch 时：
```python
next_critic_observations = self.next_privileged_observations.flatten(0, 1)
# 在 update 中再次计算这些值
```

**结论**：两种方法在数学上是等价的，但 Instinct-RL 的方案更高效。

## 与 termination observations 的集成

### 完整的数据流

```
环境端（space_mjlab）:
  env.step()
    ↓
  检测 dones
    ↓
  计算 termination_observations（来自 termination_observation_manager）
    ↓
  放在 extras["termination_observations"] 中
    ↓
  环境重置

Wrapper 端：
  获取 termination_observations
    ↓
  应用 normalizer
    ↓
  作为 info 的一部分传递给 runner

Runner 端（him_on_policy_runner.py）:
  if use_termination_obs:
      # 使用 termination_observations 替换 terminated envs 的 critic_obs
      next_critic_obs_for_bootstrap = critic_obs.clone()
      next_critic_obs_for_bootstrap[term_env_ids] = term_obs_normalized[term_env_ids]
      alg.process_env_step(..., next_critic_obs_for_bootstrap=next_critic_obs_for_bootstrap)
  else:
      # 不使用 termination observations，完全向后兼容
      alg.process_env_step(..., next_critic_obs_for_bootstrap=None)

算法端（HIMPPO）:
  bootstrap_obs = next_critic_obs_for_bootstrap if provided else next_critic_obs
  bootstrap_values = actor_critic.evaluate(bootstrap_obs)
  rewards += gamma * bootstrap_values * time_outs
```

## 关键设计决策

### 1. 不修改 Transition 类
❌ **不做**：修改 RolloutStorage.Transition 添加 next_critic_observations 字段
✅ **改做**：在 process_env_step 中直接使用和计算

**原因**：
- Transition 是 core 类，修改可能影响其他算法
- 不需要存储，因为只在当前 step 使用
- 简化代码，减少耦合

### 2. 可选参数设计
```python
def process_env_step(self, ..., next_critic_obs_for_bootstrap=None):
    bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
```

**优势**：
- 完全向后兼容
- 不提供时自动降级为标准 PPO 行为
- 类型安全（Optional[Tensor]）

### 3. PPO 和 HIMPPO 的关系
```python
# PPO：基础实现，支持 next_critic_obs_for_bootstrap（可选）
class PPO:
    def process_env_step(self, ..., next_critic_obs_for_bootstrap=None):
        # 如果提供就用，否则用默认的 next_critic_obs
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap else next_critic_obs

# HIMPPO：继承 PPO，完全相同的实现
class HIMPPO(PPO):
    def process_env_step(self, ..., next_critic_obs_for_bootstrap=None):
        # 相同逻辑，但为了清晰起见显式实现
        super().process_env_step(..., next_critic_obs_for_bootstrap)
```

**为什么 HIMPPO 要重写**：
- 虽然功能相同，但重写使意图更清晰
- 便于未来扩展 HIMPPO 特定功能
- 文档更清晰

## 性能对比

假设：
- 批量大小：2048 环境
- 更新步数：10 学习 epochs × 32 minibatches = 320 次
- Critic obs 维度：244

### 方案 A（Instinct-RL）
- 存储：每个 step 0 额外存储（计算后立即丢弃）
- 计算：每个 step 计算一次 bootstrap value = 2048 环境 × 1 计算
- **总计算**：50 steps/rollout × 2048 envs × 1 = 102,400 批量评估

### 方案 B（rsl_rl HIM）
- 存储：每个 step 存储完整 obs（244 维）× 2048 envs × 50 steps
- 计算：生成 minibatch 时再计算 = 320 次批量评估
- **总计算**：320 × 2048 = 655,360 批量评估（额外 6.4 倍）

## 总结

**Instinct-RL 的设计选择**：
- ✅ 直接在 process_env_step 中计算 bootstrap value
- ✅ 不修改 RolloutStorage（保持向后兼容）
- ✅ 可选参数设计（graceful degradation）
- ✅ 与现有架构无缝融合
- ✅ 更高效的性能

这个设计既实现了 HIM 的核心功能（termination observation bootstrapping），又保持了代码的简洁性和兼容性。
