# HIM (Hierarchical Imitation Mode) for Instinct-RL

## 概述

HIM 模块为 Instinct-RL 库添加了对 Hierarchical Imitation Mode (HIM) 的支持，基于 rsl_rl 库的实现。HIM 扩展了标准的 PPO 算法，通过在终止时使用更准确的 critic observations 来改进 value bootstrapping。

## 核心概念

在标准 PPO 中，当环境终止时，next observation 会是重置后的新状态。这可能导致 value bootstrap 不准确。HIM 通过以下方式解决：

1. **捕获终止状态**：在环境终止后、重置前捕获最后一个观测
2. **替换 next observations**：用终止观测替换已终止环境的 critic observations
3. **精确 bootstrapping**：基于正确的终止状态进行 value 估计

## 模块结构

### 1. HIMPPO 算法 (`algorithms/him_ppo.py`)

扩展 `PPO` 类，添加了对 `next_critic_obs_for_bootstrap` 参数的支持：

```python
from instinct_rl.algorithms import HIMPPO

# 在训练配置中使用
alg_cfg = {
    "class_name": "HIMPPO",
    # ... 其他 PPO 参数 ...
}
```

**主要特性**：
- 继承自标准 PPO，保持完全向后兼容
- 存储 next critic observations 用于 bootstrapping
- 支持 termination observations 替换
- 在 value loss 计算中使用正确的 bootstrap obs

### 2. HIMOnPolicyRunner (`runners/him_on_policy_runner.py`)

扩展 `OnPolicyRunner` 以支持 HIM 训练流程：

```python
from instinct_rl.runners import HIMOnPolicyRunner

runner = HIMOnPolicyRunner(env, train_cfg, log_dir="logs", device="cuda")
runner.learn(num_learning_iterations=1000)
```

**主要特性**：
- 完整的 rollout 和训练循环
- Termination observation 处理（可选）
- 支持 multi-processing 和 DDP
- 与标准 runner 兼容的日志记录

## 使用指南

### 基础配置（无 Termination Observations）

```yaml
runner:
  class_name: "HIMOnPolicyRunner"  # 使用 HIM runner
  num_steps_per_env: 24
  save_interval: 100
  use_termination_obs: false  # 禁用 termination obs

algorithm:
  class_name: "HIMPPO"  # 使用 HIMPPO 算法
  num_learning_epochs: 5
  num_mini_batches: 4
  gamma: 0.998
  lam: 0.95
  # ... 其他 PPO 参数 ...

policy:
  class_name: "ActorCritic"
  # ... 网络配置 ...
```

### 高级配置（带 Termination Observations）

```yaml
# 环境配置
env:
  class_name: "MyEnvCfg"
  # 定义 termination observations
  termination_observations:
    critic:
      terms:
        base_lin_vel: {...}
        base_ang_vel: {...}
        # ... 与 critic 相同的观测项 ...
      concatenate_terms: true

runner:
  class_name: "HIMOnPolicyRunner"
  num_steps_per_env: 24
  use_termination_obs: true  # 启用 termination obs
  
  normalizers:
    critic:
      class_name: "EmpiricalNormalization"
      # ... normalizer 配置 ...

algorithm:
  class_name: "HIMPPO"
  # ... 其他参数 ...
```

## 工作流程

启用 HIM 的训练流程：

```
1. agent.act(obs, critic_obs) → actions
2. env.step(actions) → obs, rewards, dones, infos
   ├─ 如果有环境终止：
   │  └─ 在 reset 前捕获 termination_observations
3. 应用 critic normalizer
4. 构建 next_critic_obs：
   ├─ next_critic_obs = critic_obs.clone()
   ├─ 对于终止的环境：
   │  └─ next_critic_obs[terminated_ids] = termination_critic_obs
5. HIMPPO.process_env_step(
     rewards, dones, infos, obs, critic_obs, next_critic_obs
   )
   └─ 使用 next_critic_obs 计算更准确的 bootstrap values
6. 学习阶段：
   └─ 使用存储的 next_critic_observations 计算 returns
```

## 与标准 PPO 的兼容性

HIM 模块保持了完全的向后兼容性：

1. **如果不使用 termination observations**：
   - HIMPPO 行为与标准 PPO 相同
   - next_critic_obs_for_bootstrap 默认为 None，使用 next_critic_obs
   - 可以无缝替换现有的 PPO 使用

2. **现有检查点兼容**：
   - HIMPPO 可以加载 PPO 的检查点
   - 反之亦然（在不使用新功能时）

3. **API 兼容**：
   - 与标准 OnPolicyRunner 相同的接口
   - 相同的配置参数结构

## 示例代码

### 创建环境配置

```python
from dataclasses import dataclass, field
from space_mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

@dataclass
class MyHIMEnvCfg(ManagerBasedRlEnvCfg):
    # ... 基础配置 ...
    
    termination_observations: dict[str, ObservationGroupCfg] = field(
        default_factory=lambda: {
            "critic": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                },
                concatenate_terms=True,
            )
        }
    )
```

### 训练脚本

```python
from space_mjlab.envs import ManagerBasedRlEnv
from space_mjlab.wrappers.instinct_rl import InstinctRlVecEnvWrapper
from instinct_rl.runners import HIMOnPolicyRunner

# 创建环境
cfg = MyHIMEnvCfg(...)
env = ManagerBasedRlEnv(cfg, device="cuda:0")
wrapped_env = InstinctRlVecEnvWrapper(env, use_multi_critic=False)

# 创建 runner
runner = HIMOnPolicyRunner(
    wrapped_env,
    train_cfg=train_cfg,
    log_dir="logs/him_training",
    device="cuda:0"
)

# 训练
runner.learn(num_learning_iterations=1000)
```

## 性能考虑

### 优点
- **更准确的 value 估计**：使用正确的终止状态进行 bootstrapping
- **更快的收敛**：特别是在频繁终止的任务中
- **理论上的改进**：符合无限地平线 MDP 的假设

### 缺点
- **内存开销**：为每个 step 存储额外的 next critic observations
- **计算开销**：额外的 critic evaluation（仅在有 time outs 时）

### 优化建议
- 只在有意义的 termination observations 时启用
- 使用与 critic 观测相同的组件以避免维度不匹配
- 确保 termination observations 的计算成本不过高

## 调试和监控

### 验证 Termination Observations

```python
# 检查 termination observations 是否正确计算
runner.learn(num_learning_iterations=1)
# 查看 infos 中是否包含 termination_env_ids 和 termination_observations
```

### 对比实验

建议创建两个对照组：

```python
# 配置 1：无 HIM（标准 PPO）
runner1 = OnPolicyRunner(env, train_cfg, use_termination_obs=False)

# 配置 2：有 HIM（HIMPPO）
runner2 = HIMOnPolicyRunner(env, train_cfg, use_termination_obs=True)
```

## 常见问题

**Q: HIMPPO 和标准 PPO 的主要区别是什么？**
A: 主要区别在于如何处理 bootstrapping values。HIMPPO 使用修改过的 next critic observations（可能包含 termination obs），而标准 PPO 总是使用标准的 next step observations。

**Q: 是否可以不使用 termination observations 而只用 HIMPPO？**
A: 可以。HIMPPO 在没有 termination observations 时与标准 PPO 行为相同。这在从 PPO 升级到 HIM 时很有用。

**Q: Termination observations 应该包含哪些观测项？**
A: 应该包含与 critic observations 相同的项。通常是关键的状态变量（位置、速度、角度等）。

**Q: 性能会有显著提升吗？**
A: 这取决于任务。在频繁终止的任务中通常会有改进，而在较少终止的任务中改进可能不明显。建议进行对照实验。

## 参考文献

该实现基于以下工作：
- rsl_rl 库中的 HIM 实现
- Instinct-RL 的 PPO 算法

## 许可证

SPDX-License-Identifier: BSD-3-Clause
