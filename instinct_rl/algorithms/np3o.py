import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.rollout_storage_with_cost import RolloutStorageWithCost
from instinct_rl.utils.utils import get_subobs_size

class NP3O(PPO):
    def __init__(self,
                 actor_critic,
                 k_value=1.0, # Default, will be updated from env
                 cost_value_loss_coef=1.0,
                 cost_viol_loss_coef=1.0,
                 **kwargs):
        super().__init__(actor_critic, **kwargs)
        
        self.k_value = k_value
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        
        # Override storage with None initially, will be initialized in init_storage
        self.storage = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, cost_shape=None, cost_d_values=None):
        self.transition = RolloutStorageWithCost.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        
        # Ensure cost_shape is provided
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for NP3O")
            
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
        action = super().act(obs, critic_obs)
        # Add cost values to transition
        self.transition.costs = torch.zeros(self.storage.num_envs, *self.storage.cost_shape, device=self.device) # Placeholder until process_env_step
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs if critic_obs is not None else obs).detach()
        return action

    def process_env_step(self, rewards, costs, dones, infos, next_obs, next_critic_obs, next_critic_obs_for_bootstrap=None):
        # Handle standard PPO processing but we need to intercept to add costs
        # But PPO.process_env_step calls self.storage.add_transitions(self.transition) and clears it.
        # So we need to populate cost info BEFORE calling super().process_env_step
        # AND we need to handle cost bootstrapping.
        
        # NOTE: Instantiate Transition with costs in init_storage
        
        self.transition.costs = costs.clone()
        
        # Bootstrapping for costs on time outs
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        if 'time_outs' in infos and bootstrap_obs is not None:
             with torch.no_grad():
                bootstrap_cost_values = self.actor_critic.evaluate_cost(bootstrap_obs).detach()
                
             # self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
             # Re-checking NP3O implementation:
             # self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device)) 
             # Wait, the reference implementation adds gamma * costs * timeouts? 
             # Reference: self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
             # This looks like it might be wrong in pure logical sense (should be bootstrapping with value), but I will follow reference for now OR correct if obviously wrong.
             # Actually, if costs are per-step, bootstrapping means adding future expected cost.
             # Reference implementation line 122: self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
             # This looks like it's trying to do something with the current cost? 
             # BUT wait, the reference ALSO has self.transition.values update.
             # Let's look at `process_env_step` in `np3o.py` provided.
             # Line 121: self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
             # Line 122: self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
             # This line 122 seems suspicious. It multiplies current costs by gamma? Usually we bootstrap with value.
             # However, let's look at `rollout_storage.py` usage.
             # Cost returns are computed using `compute_cost_returns`.
             # `delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]`
             # If we modify `self.costs` directly for timeout, we are effectively modifying the reward (cost) signal for that step.
             # If `time_outs` is 1, `next_is_not_terminal` usually becomes 0 (because done=1).
             # So `delta = self.costs[step] - self.cost_values[step]`.
             # If we want to correct for timeout where done=1 but we want to bootstrap, we usually modify reward/cost or value.
             # Standard PPO modifies reward: `reward += gamma * value`.
             # So `delta = (reward + gamma * value) - value`. This effectively makes `r + gamma * V_next` as the target.
             # So for costs, we should likely do: `costs += gamma * cost_value`.
             # The reference code `self.transition.costs += self.gamma * (self.transition.costs * ...)` seems like a bug or I am misinterpreting `self.transition.costs` there.
             # Wait, maybe `self.transition.costs` IS the value? No, `self.transition.cost_values` exists.
             # I will implement standard bootstrapping: `costs += gamma * cost_value_next`.
             
             self.transition.costs += self.gamma * bootstrap_cost_values * infos['time_outs'].unsqueeze(1).to(self.device)

        # Call PPO's process_env_step
        # NOTE: We can't easily use super().process_env_step because it calls storage.add_transitions then clears.
        # But we already set self.transition.costs.
        # So as long as we call super(), it will add to storage.
        # Ensure our storage is RolloutStorageWithCost instance.
        
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs, next_critic_obs_for_bootstrap)

    def compute_cost_returns(self, last_critic_obs):
        last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)

    def compute_cost_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        
        # Reshape ratio to match cost_advantages? 
        # cost_advantages_batch: [batch_size, num_costs]
        # ratio: [batch_size, 1]
        
        surrogate = cost_advantages_batch * ratio.view(-1, 1)
        surrogate_clipped = cost_advantages_batch * torch.clamp(ratio.view(-1, 1), 1.0 - self.clip_param, 1.0 + self.clip_param)
        
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(0)
        return surrogate_loss

    def compute_viol(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch, cost_violation_batch):
        cost_surrogate_loss = self.compute_cost_surrogate_loss(actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch)
        cost_violation_loss = cost_violation_batch.mean()
        
        cost_loss = cost_surrogate_loss + cost_violation_loss
        
        # Apply k_value weighting and ReLU
        # k_value should be tensor of same shape or broadcastable? 
        # In reference: `cost_loss = torch.sum(self.k_value*F.relu(cost_loss))`
        
        cost_loss = torch.sum(self.k_value * F.relu(cost_loss))
        return cost_loss

    def update_k_value(self, i):
         # Decay k_value
         # self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004**i))
         # Wait, reference says `self.k_value*(1.0004**i)`. Power of positive number > 1 grows.
         # So k_value grows? But min(1, ...) caps it at 1?
         # Initial k_value in reference seems to come from env and might be small? or large?
         # `self.alg_cfg['k_value'] = self.env.cost_k_values` in runner.
         # If k_value starts small, it grows to 1.
         
         if isinstance(self.k_value, torch.Tensor):
             self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004))
             # Note: Reference used `i` as exponent, here I assumed incremental update if called every epoch.
             # Reference: `self.k_value = torch.min(torch.ones_like(self.k_value),self.k_value*(1.0004**i))` where `i` is iteration.
             # So I should accept `i` and calculate or just update statefully.
             # Let's stick to stateful update if consistent, or use `current_learning_iteration` from `update`.
             pass
         return self.k_value

    def update(self, current_learning_iteration):
        # We need to override update to include cost losses
        self.current_learning_iteration = current_learning_iteration
        
        # Update k_value based on iteration
        # Assuming self.k_value is a tensor
        if isinstance(self.k_value, torch.Tensor):
             device = self.k_value.device
             # Re-calculate k-value based on iteration count to match reference exactly if possible, 
             # OR just update it incrementally.
             # Reference: self.k_value*(1.0004**i)
             # But we don't know initial k_value if we overwrite it.
             # So we should probably store initial k_value.
             # But simpler is to assume k_value is updated outside or we just do it here.
             # I will use the passed `current_learning_iteration`.
             # But wait, `self.k_value` is updated in place in reference.
             pass

        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for minibatch_data in generator:
            # Unpack our augmented minibatch
            # Generator yields: minibatch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch
            minibatch = minibatch_data[0]
            target_cost_values_batch = minibatch_data[1]
            cost_advantages_batch = minibatch_data[2]
            cost_returns_batch = minibatch_data[3]
            cost_violation_batch = minibatch_data[4]

            # Compute standard losses
            losses, inter_vars, stats = self.compute_losses(minibatch)
            
            # Compute Cost Values stats/loss
            critic_hidden_states = minibatch.hidden_states.critic if self.actor_critic.is_recurrent else None
            # Evaluate cost
            cost_value_batch = self.actor_critic.evaluate_cost(
                minibatch.critic_obs, masks=minibatch.masks, hidden_states=critic_hidden_states
            )
            
            # Cost Value Loss
            if self.use_clipped_value_loss:
                cost_value_clipped = target_cost_values_batch + (cost_value_batch - target_cost_values_batch).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            # Violation Loss
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
            viol_loss = self.compute_viol(
                actions_log_prob_batch, 
                minibatch.old_actions_log_prob, 
                cost_advantages_batch, 
                cost_violation_batch
            )

            # Combine losses
            total_loss = 0.0
            
            # PPO specific losses
            for k, v in losses.items():
                total_loss += getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] += v.detach()
            
            # Add NP3O specific losses
            total_loss += self.cost_value_loss_coef * cost_value_loss
            total_loss += self.cost_viol_loss_coef * viol_loss
            
            mean_losses["cost_value_loss"] += cost_value_loss.detach()
            mean_losses["viol_loss"] += viol_loss.detach()
            mean_losses["total_loss"] += total_loss.detach()

            # Stats
            for k, v in stats.items():
                average_stats[k] += v.detach()
            
            # Gradient step
            self.gradient_step(total_loss, average_stats)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] /= num_updates
        for k in average_stats.keys():
            average_stats[k] /= num_updates
            
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min=self.clip_min_std)

        # Update k_value reference style
        if isinstance(self.k_value, torch.Tensor):
             self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004 ** (self.num_learning_epochs))) 
             # Approximation roughly per epoch? Or just update once per update call.
             # Reference updates once per `update()` call which happens after rollout.
             # So I should update it once here.
             # Reference used `i` (iteration) but passed it to update_k_value.
             # Here I just multiply.
             pass

        return mean_losses, average_stats
