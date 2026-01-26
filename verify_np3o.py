
import torch
import torch.nn as nn
from instinct_rl.algorithms import NP3O
from instinct_rl.runners import OnConstraintPolicyRunner
from instinct_rl.storage.rollout_storage_with_cost import RolloutStorageWithCost

def test_imports():
    print("Imports successful!")

def test_storage_instantiation():
    try:
        storage = RolloutStorageWithCost(
            num_envs=2,
            num_transitions_per_env=10,
            obs_shape=(5,),
            critic_obs_shape=(5,),
            actions_shape=(2,),
            cost_shape=(1,),
            cost_d_values=torch.tensor([0.1])
        )
        print("Storage instantiation successful!")
        assert hasattr(storage, 'costs')
        assert hasattr(storage, 'cost_values')
    except Exception as e:
        print(f"Storage instantiation failed: {e}")
        raise e

def test_algorithm_instantiation():
    # Mock ActorCritic
    class MockActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_recurrent = False
            self.actor = nn.Linear(5, 2)
            self.critic = nn.Linear(5, 1)
            self.action_mean = torch.zeros(2)
            self.action_std = torch.ones(2)
        def act(self, obs, critic_obs=None):
            return torch.zeros(obs.shape[0], 2)
        def evaluate(self, obs):
            return torch.zeros(obs.shape[0], 1)
        # Mock evaluate_cost needed for NP3O
        def evaluate_cost(self, obs, masks=None, hidden_states=None):
            return torch.zeros(obs.shape[0], 1)
        def get_actions_log_prob(self, actions):
            return torch.zeros(actions.shape[0], 1)
        def reset(self, dones):
            pass

    actor_critic = MockActorCritic()
    
    try:
        alg = NP3O(actor_critic, k_value=torch.tensor(1.0))
        # Init storage
        alg.init_storage(
            num_envs=2,
            num_transitions_per_env=10,
            obs_format={"policy": {"base": (5,)}, "critic": {"base": (5,)}},
            num_actions=2,
            cost_shape=(1,),
            cost_d_values=torch.tensor([0.1])
        )
        print("Algorithm instantiation successful!")
        
        # Test act
        obs = torch.randn(2, 5)
        alg.act(obs, obs)
        print("Algorithm act successful!")
        
        # Test update (dry run without data in storage might fail or return 0 losses if empty generator)
        # We won't test full update here as it requires data filling.
        
    except Exception as e:
        print(f"Algorithm instantiation failed: {e}")
        raise e

if __name__ == "__main__":
    test_imports()
    test_storage_instantiation()
    test_algorithm_instantiation()
