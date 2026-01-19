# Termination Observation Usage Guide

## 概述

此功能允许在环境终止后、重置前捕获观测，并用这些观测替换 critic 的 next observation，以提供更准确的 bootstrapping 值估计。

## 功能说明

在标准的 PPO 算法中，当环境终止时，next observation 会是重置后的新状态，这会导致 value bootstrapping 不准确。此功能通过以下方式解决这个问题：

1. **捕获终止观测**：在环境 step 中，termination 检测后、reset 前捕获观测
2. **替换 next observation**：用终止观测替换已终止环境的 critic observation
3. **精确 bootstrapping**：使用正确的终止状态来估计 value，提高学习效率

## 环境配置

### 1. 在环境配置中添加 `termination_observations`

```python
from dataclasses import dataclass, field
from space_mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

@dataclass
class MyEnvCfg(ManagerBasedRlEnvCfg):
    # ... 其他配置 ...
    
    # 添加 termination observations 配置
    termination_observations: dict[str, ObservationGroupCfg] = field(
        default_factory=lambda: {
            "critic": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                    # ... 添加所有 critic 需要的观测项 ...
                },
                concatenate_terms=True,
            )
        }
    )
```

**注意**：
- `termination_observations` 中的观测组名称应该与正常观测中的组名称匹配（如 "critic"）
- 观测项应该与正常 critic observations 相同，以确保维度匹配

### 2. 确保环境返回 termination observations

环境的 `step()` 方法会自动在 `extras` 中提供：
- `extras["termination_env_ids"]`: 终止的环境 ID
- `extras["termination_observations"]`: 终止时的观测字典

## Runner 配置

在训练配置文件中启用此功能：

```python
runner_cfg = {
    "algorithm": {...},
    "policy": {...},
    "num_steps_per_env": 24,
    "save_interval": 100,
    
    # 启用 termination observation 功能
    "use_termination_obs": True,  # 设置为 True 启用，False 禁用（默认）
    
    "normalizers": {
        "critic": {
            "class_name": "EmpiricalNormalization",
            # ... normalizer 配置 ...
        }
    },
    # ... 其他配置 ...
}
```

## 工作流程

启用后，训练流程如下：

```
1. agent.act(obs, critic_obs) → actions
2. env.step(actions) → obs, rewards, dones, infos
   └─ 如果有环境终止：
      └─ 在 reset 前捕获 termination_observations
3. 应用 normalizer 到 critic_obs
4. 构建 next_critic_obs:
   - next_critic_obs = critic_obs.clone()
   - 对于终止的环境：
     └─ next_critic_obs[terminated_ids] = termination_critic_obs
5. PPO.process_env_step(..., next_critic_obs_for_bootstrap=next_critic_obs)
   └─ 使用 next_critic_obs 计算 bootstrap values
```

## 关键代码片段

### Runner 中的处理逻辑

```python
def rollout_step(self, obs, critic_obs):
    actions = self.alg.act(obs, critic_obs)
    obs, rewards, dones, infos = self.env.step(actions)
    
    # ... 获取和规范化 critic_obs ...
    
    # 构建 next_critic_obs 用于 bootstrapping
    next_critic_obs = None
    if self.use_termination_obs and critic_obs is not None:
        termination_env_ids = infos.get("termination_env_ids", torch.tensor([]))
        termination_obs = infos.get("termination_observations", {})
        
        if len(termination_env_ids) > 0 and len(termination_obs) > 0:
            next_critic_obs = critic_obs.clone().detach()
            term_critic_obs = termination_obs.get("critic", None)
            if term_critic_obs is not None:
                # 应用 normalizer
                if "critic" in self.normalizers:
                    term_critic_obs = self.normalizers["critic"](term_critic_obs)
                # 替换终止环境的观测
                next_critic_obs[termination_env_ids] = term_critic_obs.clone().detach()
    
    if next_critic_obs is None:
        next_critic_obs = critic_obs
    
    self.alg.process_env_step(rewards, dones, infos, obs, critic_obs, next_critic_obs)
```

### PPO 算法中的处理

```python
def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs, 
                     next_critic_obs_for_bootstrap=None):
    # ... 计算 rewards 和 auxiliary rewards ...
    
    # Bootstrapping on time outs
    bootstrap_obs = (next_critic_obs_for_bootstrap 
                     if next_critic_obs_for_bootstrap is not None 
                     else next_critic_obs)
    
    if "time_outs" in infos and bootstrap_obs is not None:
        with torch.no_grad():
            bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
        self.transition.rewards += (
            self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1)
        )
```

## 性能考虑

- **内存开销**：为终止的环境额外存储一份观测（通常每步只有少量环境终止）
- **计算开销**：额外的 critic evaluation 用于 bootstrapping（仅在有 time outs 时）
- **训练效果**：理论上提供更准确的 value 估计，特别是在频繁终止的任务中

## 调试技巧

1. **验证观测维度**：确保 `termination_observations` 与正常 `critic` 观测维度一致
2. **检查终止率**：使用 `len(termination_env_ids)` 监控每步有多少环境终止
3. **对比实验**：设置 `use_termination_obs: False` 作为对照组

## 适用场景

此功能特别适用于：
- 存在明确失败条件的任务（如机器人跌倒、超出边界等）
- Episode length 较短且频繁重置的环境
- 需要精确 value 估计的任务

## 兼容性

- ✅ 支持单 critic 和多 critic 模式
- ✅ 支持 observation normalization
- ✅ 支持 multi-processing training
- ✅ 向后兼容（默认禁用）
