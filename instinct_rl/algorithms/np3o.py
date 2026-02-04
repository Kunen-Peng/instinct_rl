"""
NP3O (Penalized Proximal Policy Optimization) Algorithm Implementation.

Based on the P3O paper, this algorithm transforms constrained RL into unconstrained
optimization using an exact penalty function:

    L^{P3O}(θ) = L_R^{CLIP}(θ) + κ · Σ_i max{0, L_{C_i}^{CLIP}(θ)}

Key Components:
1. Dual Value Networks: Reward Critic + Cost Critic
2. Cost Advantage calculation using GAE
3. Cost Violation term with proper normalization
4. ReLU truncation to ensure penalties only when violating constraints
5. Adaptive κ (k_value) scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.rollout_storage_with_cost import RolloutStorageWithCost
from instinct_rl.utils.utils import get_subobs_size


class NP3O(PPO):
    """
    NP3O: Penalized Proximal Policy Optimization for Constrained RL.
    
    Extends PPO with:
    - Cost Critic network for estimating expected cumulative cost
    - Cost advantage and violation term computation  
    - Penalty-based constraint handling with ReLU truncation
    """
    
    def __init__(
        self,
        actor_critic,
        k_value=1.0,  # Penalty coefficient, can be tensor for multiple constraints
        k_value_max=1.0,  # Maximum k_value cap
        k_value_growth_rate=1.0004,  # Growth rate per update
        k_warmup_iterations=100,  # κ stays at initial value during warmup
        adaptive_alpha=0.02,  # Threshold relaxation factor for adaptive scheduling
        cost_value_loss_coef=1.0,
        cost_viol_loss_coef=1.0,
        **kwargs
    ):
        super().__init__(actor_critic, **kwargs)
        
        # Store initial k_value for reference
        self._initial_k_value = k_value
        self.k_value = k_value
        self.k_value_max = k_value_max
        self.k_value_growth_rate = k_value_growth_rate
        self.k_warmup_iterations = k_warmup_iterations
        self.adaptive_alpha = adaptive_alpha
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        
        # Storage will be initialized in init_storage
        self.storage = None

    def init_storage(
        self, 
        num_envs, 
        num_transitions_per_env, 
        obs_format, 
        num_actions, 
        num_rewards=1, 
        cost_shape=None, 
        cost_d_values=None
    ):
        """
        Initialize rollout storage with cost tracking.
        
        Args:
            cost_shape: Tuple indicating shape of cost vector, e.g., (num_costs,)
            cost_d_values: Cost limits/thresholds, tensor of shape cost_shape
        """
        self.transition = RolloutStorageWithCost.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for NP3O")
        
        # Convert k_value to tensor if needed, matching cost dimensions
        if not isinstance(self.k_value, torch.Tensor):
            self.k_value = torch.tensor(
                [self.k_value] * cost_shape[0] if isinstance(cost_shape, tuple) else [self.k_value],
                dtype=torch.float32,
                device=self.device
            )
        else:
            self.k_value = self.k_value.to(self.device)
            
        self.storage = RolloutStorageWithCost(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        """
        Sample action and record cost values for current state.
        
        Note: cost_values are computed here (similar to values in PPO),
        but costs are filled in process_env_step() from environment output
        (similar to rewards in PPO).
        """
        action = super().act(obs, critic_obs)
        
        # Evaluate cost values using Cost Critic (analogous to value estimation)
        with torch.no_grad():
            critic_input = critic_obs if critic_obs is not None else obs
            self.transition.cost_values = self.actor_critic.evaluate_cost(critic_input).detach()
        
        # Note: self.transition.costs is set in process_env_step(), not here
        # This is the same pattern as rewards in PPO
        
        return action

    def process_env_step(
        self, 
        rewards, 
        costs, 
        dones, 
        infos, 
        next_obs, 
        next_critic_obs, 
        next_critic_obs_for_bootstrap=None
    ):
        """
        Process environment step with cost information.
        
        Handles:
        1. Recording costs from environment
        2. Bootstrapping cost values for truncated episodes (time_outs)
        """
        # Record costs
        self.transition.costs = costs.clone()
        
        # Handle timeout bootstrapping for costs
        # When episode is truncated (time_out=1, done=1), we need to add
        # bootstrapped future cost value to current cost
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        
        if 'time_outs' in infos and bootstrap_obs is not None:
            time_outs = infos['time_outs'].to(self.device)
            
            with torch.no_grad():
                bootstrap_cost_values = self.actor_critic.evaluate_cost(bootstrap_obs).detach()
            
            # Bootstrap: cost += gamma * V_cost(next_state) for timed-out episodes
            # This corrects the TD target when done=True but episode was just truncated
            self.transition.costs = self.transition.costs + self.gamma * bootstrap_cost_values * time_outs.unsqueeze(-1)

        # Call parent's process_env_step (handles rewards, values, storage)
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs, next_critic_obs_for_bootstrap)

    def compute_cost_returns(self, last_critic_obs):
        """
        Compute cost returns and advantages using GAE.
        Also computes normalized cost violation term.
        """
        with torch.no_grad():
            last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)

    def compute_cost_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch):
        """
        Compute clipped surrogate loss for cost objective.
        
        Similar to PPO's reward surrogate, but for cost minimization:
        - Uses max instead of min (we want to minimize cost, so maximize negative cost)
        - Applies clipping to prevent large policy updates
        
        Returns:
            cost_surrogate_loss: Shape (num_costs,) - per-constraint surrogate losses
        """
        # Compute probability ratio
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        
        # Expand ratio to match cost dimensions: [batch_size] -> [batch_size, 1]
        ratio_expanded = ratio.view(-1, 1)
        
        # Cost surrogate (note: we want to minimize cost, so use positive advantage)
        surrogate = cost_advantages_batch * ratio_expanded
        surrogate_clipped = cost_advantages_batch * torch.clamp(
            ratio_expanded, 
            1.0 - self.clip_param, 
            1.0 + self.clip_param
        )
        
        # Take max (pessimistic for cost minimization) and mean over batch
        # Returns shape: (num_costs,)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(dim=0)
        
        return surrogate_loss

    def compute_viol(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch, cost_violation_batch):
        """
        Compute the penalized constraint violation loss.
        
        P3O Loss = κ · ReLU(Cost_Surrogate + Cost_Violation)
        
        The ReLU ensures:
        - Gradient is 0 when policy is safe (cost < limit)
        - Algorithm degrades to standard PPO when all constraints satisfied
        
        Args:
            cost_advantages_batch: Normalized cost advantages, shape [batch, num_costs]
            cost_violation_batch: Normalized violation term, shape [batch, num_costs]
            
        Returns:
            Scalar penalty loss
        """
        # Compute cost surrogate loss per constraint, shape: (num_costs,)
        cost_surrogate_loss = self.compute_cost_surrogate_loss(
            actions_log_prob_batch, 
            old_actions_log_prob_batch, 
            cost_advantages_batch
        )
        
        # Compute mean violation per constraint, shape: (num_costs,)
        cost_violation_loss = cost_violation_batch.mean(dim=0)
        
        # Combined cost loss per constraint
        cost_loss = cost_surrogate_loss + cost_violation_loss
        
        # Apply ReLU: only penalize when violating or likely to violate
        # Then weight by k_value and sum across constraints
        cost_loss = torch.sum(self.k_value * F.relu(cost_loss))
        
        return cost_loss

    def update_adaptive_constraints(self, current_mean_costs):
        """
        Update active constraint thresholds based on current rollout performance.
        
        Implements Kim et al. (2024) adaptive threshold strategy:
        d_new = max(d_target, J_current - alpha * d_target)
        
        This creates a "moving goalpost" that's slightly better than current performance,
        encouraging gradual improvement while avoiding overly aggressive penalties.
        
        Args:
            current_mean_costs: Mean cost per constraint from current rollout, shape (num_costs,)
            
        Returns:
            Updated active threshold values, or None if thresholds not configured
        """
        target_d = self.storage.target_cost_d_values
        if target_d is None:
            return None
        
        # Ensure tensors are on same device
        if current_mean_costs.device != target_d.device:
            current_mean_costs = current_mean_costs.to(target_d.device)
        
        # Calculate margin based on target threshold
        margin = self.adaptive_alpha * target_d
        
        # New threshold: max(target, current - margin)
        # This ensures the threshold is never below target, but can be relaxed above target
        new_active_d = torch.max(target_d, current_mean_costs - margin)
        
        # Update storage with new active thresholds
        self.storage.update_active_d_values(new_active_d)
        
        return new_active_d

    def update_k_value(self):
        """
        Update penalty coefficient κ using exponential growth schedule.
        
        The k_value grows from initial value towards k_value_max to enforce
        stricter constraint satisfaction as training progresses.
        
        Includes warmup period where k_value stays fixed to allow Cost Critic
        to train before applying strong penalties.
        """
        # Skip growth during warmup period
        if self.current_learning_iteration < self.k_warmup_iterations:
            return
            
        if isinstance(self.k_value, torch.Tensor):
            # Exponential growth with cap - use in-place operations to avoid memory allocation
            self.k_value.mul_(self.k_value_growth_rate)
            self.k_value.clamp_(max=self.k_value_max)
        elif isinstance(self.k_value, (int, float)):
            self.k_value = min(self.k_value_max, self.k_value * self.k_value_growth_rate)

    def update(self, current_learning_iteration):
        """
        Perform NP3O policy update.
        
        Extends PPO update with:
        1. Cost value loss (train Cost Critic)
        2. Violation loss (penalized constraint objective)
        3. K-value scheduling
        """
        self.current_learning_iteration = current_learning_iteration
        
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        
        # Select appropriate batch generator
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            
        for minibatch_data in generator:
            # Unpack augmented minibatch
            minibatch = minibatch_data[0]
            target_cost_values_batch = minibatch_data[1]
            cost_advantages_batch = minibatch_data[2]
            cost_returns_batch = minibatch_data[3]
            cost_violation_batch = minibatch_data[4]

            # Step 1: Compute standard PPO losses (surrogate, value, entropy)
            losses, inter_vars, stats = self.compute_losses(minibatch)
            
            # Step 2: Compute Cost Critic loss
            critic_hidden_states = minibatch.hidden_states.critic if self.actor_critic.is_recurrent else None
            cost_value_batch = self.actor_critic.evaluate_cost(
                minibatch.critic_obs, 
                masks=minibatch.masks, 
                hidden_states=critic_hidden_states
            )
            
            # Cost Value Loss with optional clipping
            if self.use_clipped_value_loss:
                cost_value_clipped = target_cost_values_batch + (
                    cost_value_batch - target_cost_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            # Step 3: Compute Violation Loss (P3O core)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
            viol_loss = self.compute_viol(
                actions_log_prob_batch, 
                minibatch.old_actions_log_prob, 
                cost_advantages_batch, 
                cost_violation_batch
            )

            # Step 4: Combine all losses
            total_loss = 0.0
            
            # PPO losses (surrogate, value, entropy)
            for k, v in losses.items():
                coef = getattr(self, k + "_coef", 1.0)
                total_loss = total_loss + coef * v
                mean_losses[k] += v.detach()
            
            # NP3O specific losses
            total_loss = total_loss + self.cost_value_loss_coef * cost_value_loss
            total_loss = total_loss + self.cost_viol_loss_coef * viol_loss
            
            mean_losses["cost_value_loss"] += cost_value_loss.detach()
            mean_losses["viol_loss"] += viol_loss.detach()
            mean_losses["total_loss"] += total_loss.detach()

            # Collect stats
            for k, v in stats.items():
                average_stats[k] += v.detach()
            
            # Step 5: Gradient step
            self.gradient_step(total_loss, average_stats)

        # Compute mean losses over all updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] /= num_updates
        for k in average_stats.keys():
            average_stats[k] /= num_updates
            
        # Clear storage
        self.storage.clear()
        
        # Clip action std if needed
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min=self.clip_min_std)

        # Step 6: Update penalty coefficient κ
        self.update_k_value()
        
        # Add k_value to stats for logging
        if isinstance(self.k_value, torch.Tensor):
            average_stats["k_value_mean"] = self.k_value.mean()
        else:
            average_stats["k_value"] = self.k_value

        return mean_losses, average_stats

    def state_dict(self):
        """Save algorithm state including k_value."""
        state = super().state_dict()
        state["k_value"] = self.k_value
        return state
    
    def load_state_dict(self, state_dict):
        """Load algorithm state including k_value."""
        if "k_value" in state_dict:
            self.k_value = state_dict.pop("k_value")
            if isinstance(self.k_value, torch.Tensor):
                self.k_value = self.k_value.to(self.device)
        super().load_state_dict(state_dict)
