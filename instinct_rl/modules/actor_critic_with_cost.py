
import torch
import torch.nn as nn
from typing import Dict, List
from instinct_rl.modules.actor_critic import ActorCritic

class ActorCriticWithCost(ActorCritic):
    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions: int,
        num_rewards: int = 1,
        num_costs: int = 0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__(
            obs_format=obs_format,
            num_actions=num_actions,
            num_rewards=num_rewards,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs
        )

        self.num_costs = num_costs
        if self.num_costs > 0:
            print(f"Building Cost Critic with {self.num_costs} outputs")
            self.cost_critic = self._build_critic(num_values=self.num_costs)
            print(f"Cost Critic MLP: {self.cost_critic}")
        else:
            print("Warning: ActorCriticWithCost initialized with num_costs=0")

    def evaluate_cost(self, critic_observations, **kwargs):
        if hasattr(self, "cost_critic"):
            return self.cost_critic(critic_observations)
        else:
            return None
