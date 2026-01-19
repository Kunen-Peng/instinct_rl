#!/usr/bin/env python3
"""
验证脚本：检验 HIMPPO 和 HIMOnPolicyRunner 的正确性

这个脚本验证：
1. Transition 类不包含 next_critic_observations
2. RolloutStorage 可以正常添加 transitions
3. process_env_step 的参数签名正确
4. next_critic_obs_for_bootstrap 参数正确处理
"""

import sys
import torch
from pathlib import Path

# 添加 instinct_rl 到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_transition_structure():
    """验证 Transition 类的结构"""
    from instinct_rl.storage import RolloutStorage
    
    print("=" * 60)
    print("TEST 1: Transition 类结构")
    print("=" * 60)
    
    transition = RolloutStorage.Transition()
    
    # 获取 Transition 的所有属性
    attrs = {k: v for k, v in transition.__dict__.items() if not k.startswith('_')}
    
    print(f"✓ Transition 属性数量: {len(attrs)}")
    print(f"✓ Transition 属性列表: {list(attrs.keys())}")
    
    # 验证不包含 next_critic_observations
    if 'next_critic_observations' in attrs:
        print("❌ 错误: Transition 包含 next_critic_observations")
        return False
    else:
        print("✓ 正确: Transition 不包含 next_critic_observations")
    
    print()
    return True


def test_rollout_storage():
    """验证 RolloutStorage 可以正常工作"""
    from instinct_rl.storage import RolloutStorage
    
    print("=" * 60)
    print("TEST 2: RolloutStorage 功能")
    print("=" * 60)
    
    # 创建 storage
    num_envs = 4
    num_transitions = 8
    obs_shape = (10,)
    critic_obs_shape = (12,)
    actions_shape = (3,)
    
    storage = RolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs_shape=obs_shape,
        critic_obs_shape=critic_obs_shape,
        actions_shape=actions_shape,
        device='cpu'
    )
    
    print(f"✓ 创建 RolloutStorage 成功")
    print(f"  - num_envs: {num_envs}")
    print(f"  - num_transitions: {num_transitions}")
    print(f"  - obs_shape: {obs_shape}")
    print(f"  - critic_obs_shape: {critic_obs_shape}")
    
    # 创建 transition 并添加到 storage
    try:
        transition = RolloutStorage.Transition()
        transition.observations = torch.randn(num_envs, *obs_shape)
        transition.critic_observations = torch.randn(num_envs, *critic_obs_shape)
        transition.actions = torch.randn(num_envs, *actions_shape)
        transition.rewards = torch.randn(num_envs, 1)
        transition.dones = torch.zeros(num_envs, 1, dtype=torch.uint8)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.action_mean = torch.randn(num_envs, *actions_shape)
        transition.action_sigma = torch.randn(num_envs, *actions_shape)
        
        storage.add_transitions(transition)
        print(f"✓ add_transitions 成功")
    except Exception as e:
        print(f"❌ add_transitions 失败: {e}")
        return False
    
    print()
    return True


def test_ppo_signature():
    """验证 PPO 和 HIMPPO 的 process_env_step 签名"""
    from instinct_rl.algorithms import PPO, HIMPPO
    import inspect
    
    print("=" * 60)
    print("TEST 3: process_env_step 函数签名")
    print("=" * 60)
    
    # 检查 PPO 的签名
    ppo_sig = inspect.signature(PPO.process_env_step)
    print(f"✓ PPO.process_env_step 参数: {list(ppo_sig.parameters.keys())}")
    
    ppo_params = list(ppo_sig.parameters.keys())
    expected_params = ['self', 'rewards', 'dones', 'infos', 'next_obs', 'next_critic_obs', 'next_critic_obs_for_bootstrap']
    
    if ppo_params == expected_params:
        print(f"✓ 参数列表正确")
    else:
        print(f"❌ 参数列表不匹配")
        print(f"  期望: {expected_params}")
        print(f"  实际: {ppo_params}")
        return False
    
    # 检查 next_critic_obs_for_bootstrap 是否有默认值
    param = ppo_sig.parameters['next_critic_obs_for_bootstrap']
    if param.default is None:
        print(f"✓ next_critic_obs_for_bootstrap 默认值为 None")
    else:
        print(f"❌ next_critic_obs_for_bootstrap 默认值错误: {param.default}")
        return False
    
    # 检查 HIMPPO 的签名
    himppo_sig = inspect.signature(HIMPPO.process_env_step)
    print(f"✓ HIMPPO.process_env_step 参数: {list(himppo_sig.parameters.keys())}")
    
    if list(himppo_sig.parameters.keys()) == expected_params:
        print(f"✓ HIMPPO 参数列表正确")
    else:
        print(f"❌ HIMPPO 参数列表不匹配")
        return False
    
    print()
    return True


def test_parameter_handling():
    """验证 next_critic_obs_for_bootstrap 的处理"""
    
    print("=" * 60)
    print("TEST 4: 参数传递验证")
    print("=" * 60)
    
    # 通过代码检查验证 process_env_step 的行为
    from instinct_rl.algorithms import PPO, HIMPPO
    import inspect
    
    # 检查两个类的实现源码
    ppo_source = inspect.getsource(PPO.process_env_step)
    himppo_source = inspect.getsource(HIMPPO.process_env_step)
    
    # 验证关键的 bootstrap 逻辑
    if "next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs" in ppo_source:
        print("✓ PPO 正确处理 next_critic_obs_for_bootstrap 参数")
    else:
        print("❌ PPO 没有正确处理参数")
        return False
    
    if "next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs" in himppo_source:
        print("✓ HIMPPO 正确处理 next_critic_obs_for_bootstrap 参数")
    else:
        print("❌ HIMPPO 没有正确处理参数")
        return False
    
    # 验证 storage.add_transitions 被调用
    if "self.storage.add_transitions(self.transition)" in ppo_source:
        print("✓ PPO 正确调用 storage.add_transitions")
    else:
        print("❌ PPO 没有调用 storage.add_transitions")
        return False
    
    # 验证没有试图存储 next_critic_observations
    if "self.transition.next_critic_observations" in ppo_source:
        print("❌ PPO 试图存储 next_critic_observations（这是错的）")
        return False
    else:
        print("✓ PPO 不试图存储 next_critic_observations")
    
    if "self.transition.next_critic_observations" in himppo_source:
        print("❌ HIMPPO 试图存储 next_critic_observations（这是错的）")
        return False
    else:
        print("✓ HIMPPO 不试图存储 next_critic_observations")
    
    print()
    return True


def test_him_imports():
    """验证 HIMPPO 和 HIMOnPolicyRunner 可以导入"""
    print("=" * 60)
    print("TEST 5: 模块导入")
    print("=" * 60)
    
    try:
        from instinct_rl.algorithms import HIMPPO
        print("✓ HIMPPO 导入成功")
    except ImportError as e:
        print(f"❌ HIMPPO 导入失败: {e}")
        return False
    
    try:
        from instinct_rl.runners import HIMOnPolicyRunner
        print("✓ HIMOnPolicyRunner 导入成功")
    except ImportError as e:
        print(f"❌ HIMOnPolicyRunner 导入失败: {e}")
        return False
    
    print()
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("HIMPPO 正确性验证测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("Transition 结构", test_transition_structure),
        ("RolloutStorage 功能", test_rollout_storage),
        ("PPO 签名", test_ppo_signature),
        ("参数处理", test_parameter_handling),
        ("模块导入", test_him_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！")
        print("\n核心发现:")
        print("1. ✓ Transition 不包含 next_critic_observations（这是正确的）")
        print("2. ✓ RolloutStorage 无需修改")
        print("3. ✓ process_env_step 正确接受 next_critic_obs_for_bootstrap 参数")
        print("4. ✓ 参数在 process_env_step 中立即处理（不存储）")
        print("5. ✓ HIMPPO 和 HIMOnPolicyRunner 正确导出")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
