"""
Actor-Critic module with additional Cost Critic for Constrained RL (P3O/NP3O).

This module extends the standard ActorCritic to include a Cost Critic network
that estimates the expected cumulative cost for each constraint type.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.cost_critic import (
    MultiHeadCostCritic,
    VectorCostCritic,
    resolve_multi_head_dims,
)


class ActorCriticWithCost(ActorCritic):
    """
    Actor-Critic with Cost Value Function for Constrained RL.
    
    Extends ActorCritic with:
    - Cost Critic: Estimates expected cumulative cost V_C(s) for each constraint
    - Supports multiple cost types (e.g., collision, velocity limit)
    
    Architecture:
        Actor: obs -> hidden -> action distribution
        Reward Critic: critic_obs -> hidden -> V_R(s)
        Cost Critic: critic_obs -> hidden -> V_C(s) [num_costs outputs]
    """
    
    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions: int,
        num_rewards: int = 1,
        num_costs: int = 0,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        cost_critic_hidden_dims: Optional[List[int]] = None,  # Defaults to critic_hidden_dims if None
        cost_critic_type: str = "vector_head",
        cost_backbone_hidden_dims: Optional[List[int]] = None,
        cost_head_hidden_dims: Optional[List[int]] = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        """
        Initialize ActorCriticWithCost.
        
        Args:
            obs_format: Dictionary specifying observation structure
            num_actions: Number of action dimensions
            num_rewards: Number of reward signals (for multi-objective)
            num_costs: Number of cost/constraint types
            actor_hidden_dims: Hidden layer sizes for actor
            critic_hidden_dims: Hidden layer sizes for reward critic
            cost_critic_hidden_dims: Hidden layer sizes for cost critic (defaults to critic_hidden_dims)
            cost_critic_type: Cost-critic architecture, one of {"vector_head", "multi_head"}
            cost_backbone_hidden_dims: Shared backbone sizes for multi-head cost critic
            cost_head_hidden_dims: Per-head MLP sizes for multi-head cost critic
            activation: Activation function name
            init_noise_std: Initial action noise std
        """
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
        
        # Use same architecture as reward critic if not specified
        if cost_critic_hidden_dims is None:
            cost_critic_hidden_dims = critic_hidden_dims
        self.cost_critic_hidden_dims = cost_critic_hidden_dims
        self.cost_critic_type = cost_critic_type
        self.cost_backbone_hidden_dims, self.cost_head_hidden_dims = resolve_multi_head_dims(
            default_hidden_dims=self.cost_critic_hidden_dims,
            backbone_hidden_dims=cost_backbone_hidden_dims,
            head_hidden_dims=cost_head_hidden_dims,
        )
        
        if self.num_costs > 0:
            print(
                f"[ActorCriticWithCost] Building {self.cost_critic_type} cost critic with {self.num_costs} cost types"
            )
            self.cost_critic = self._build_cost_critic(num_costs=self.num_costs)
            print(f"[ActorCriticWithCost] Cost Critic architecture: {self.cost_critic}")
        else:
            print("[ActorCriticWithCost] Warning: Initialized with num_costs=0, no cost critic built")
            self.cost_critic = None

    def _build_cost_critic(self, num_costs: int) -> nn.Module:
        """
        Build the cost critic network.
        """
        if self.cost_critic_type == "vector_head":
            return VectorCostCritic(
                input_dim=self.mlp_input_dim_c,
                hidden_dims=self.cost_critic_hidden_dims,
                num_costs=num_costs,
                activation_name=self.activation,
            )
        if self.cost_critic_type == "multi_head":
            return MultiHeadCostCritic(
                input_dim=self.mlp_input_dim_c,
                num_costs=num_costs,
                activation_name=self.activation,
                backbone_hidden_dims=self.cost_backbone_hidden_dims,
                head_hidden_dims=self.cost_head_hidden_dims,
            )
        raise ValueError(f"Unsupported cost_critic_type: {self.cost_critic_type}")

    def evaluate_cost(
        self, 
        critic_observations: torch.Tensor, 
        masks: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Evaluate cost value function V_C(s).
        
        Args:
            critic_observations: Observations for critic input
            masks: Optional masks for recurrent processing
            hidden_states: Optional hidden states for recurrent critic
            
        Returns:
            Cost values tensor of shape (batch_size, num_costs), or None if no cost critic
        """
        if self.cost_critic is None:
            return None
            
        # Note: Current implementation assumes MLP cost critic (non-recurrent)
        # For recurrent cost critic, would need to handle hidden states
        return self.cost_critic(critic_observations)

    def reset(self, dones=None):
        """Reset internal states (if any)."""
        super().reset(dones)
        # Cost critic is currently MLP, no internal state to reset
        # If recurrent cost critic is added, handle reset here

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load checkpoints with backward compatibility for older vector cost-critic keys."""
        remapped_state_dict = dict(state_dict)
        remapped_keys = {}

        for key in list(remapped_state_dict.keys()):
            if key.startswith("cost_critic.") and not key.startswith("cost_critic.model."):
                suffix = key[len("cost_critic.") :]
                if suffix and suffix[0].isdigit():
                    remapped_keys[key] = f"cost_critic.model.{suffix}"

        for old_key, new_key in remapped_keys.items():
            remapped_state_dict[new_key] = remapped_state_dict.pop(old_key)

        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)
