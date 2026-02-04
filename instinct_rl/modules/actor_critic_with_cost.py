"""
Actor-Critic module with additional Cost Critic for Constrained RL (P3O/NP3O).

This module extends the standard ActorCritic to include a Cost Critic network
that estimates the expected cumulative cost for each constraint type.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from instinct_rl.modules.actor_critic import ActorCritic


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
        
        if self.num_costs > 0:
            print(f"[ActorCriticWithCost] Building Cost Critic with {self.num_costs} cost types")
            # Build cost critic with same input as reward critic
            self.cost_critic = self._build_cost_critic(num_costs=self.num_costs)
            print(f"[ActorCriticWithCost] Cost Critic architecture: {self.cost_critic}")
        else:
            print("[ActorCriticWithCost] Warning: Initialized with num_costs=0, no cost critic built")
            self.cost_critic = None

    def _build_cost_critic(self, num_costs: int) -> nn.Sequential:
        """
        Build the cost critic network.
        
        Uses the same input dimension as the reward critic (mlp_input_dim_c).
        Applies orthogonal initialization for stable training.
        
        Args:
            num_costs: Number of cost values to predict
            
        Returns:
            Cost critic network as nn.Sequential
        """
        import numpy as np
        from instinct_rl.modules.actor_critic import get_activation
        
        activation = get_activation(self.activation)
        hidden_dims = self.cost_critic_hidden_dims
        
        layers = []
        layers.append(nn.Linear(self.mlp_input_dim_c, hidden_dims[0]))
        layers.append(activation)
        
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                # Output layer
                layers.append(nn.Linear(hidden_dims[l], num_costs))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        
        model = nn.Sequential(*layers)
        
        # --- Weight Initialization for Stability ---
        # Orthogonal init with appropriate gains for each layer
        output_layer = None
        for i, m in enumerate(model.modules()):
            if isinstance(m, nn.Linear):
                # Orthogonal initialization (better for RL)
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                output_layer = m  # Track last linear layer
        
        # Small gain for output layer to ensure near-zero initial predictions
        # This prevents large initial cost values causing gradient explosion
        if output_layer is not None:
            nn.init.orthogonal_(output_layer.weight, gain=0.01)
            nn.init.constant_(output_layer.bias, 0.0)
                
        return model

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
