from collections import namedtuple
import torch
from instinct_rl.storage import RolloutStorage
from instinct_rl.utils import split_and_pad_trajectories

class RolloutStorageWithCost(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.costs = None
            self.cost_values = None

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, cost_shape, cost_d_values, device='cpu'):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, device=device)
        self.cost_shape = cost_shape
        
        # Target threshold (final goal) - never changes during training
        self.target_cost_d_values = cost_d_values.clone() if cost_d_values is not None else None
        # Active threshold (used for violation calculation) - may be relaxed for adaptive scheduling
        self.active_cost_d_values = cost_d_values.clone() if cost_d_values is not None else None
        # Backward compatibility alias
        self.cost_d_values = self.active_cost_d_values

        self.costs = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_violation = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        # Add cost specific data
        self.costs[self.step].copy_(transition.costs)
        self.cost_values[self.step].copy_(transition.cost_values)
        
        # Call parent to add standard data
        super().add_transitions(transition)

    def compute_cost_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]

        # Compute and normalize the cost advantages
        self.cost_advantages = self.cost_returns - self.cost_values
        cost_adv_mean = self.cost_advantages.view(self.num_envs*self.num_transitions_per_env, -1).mean(0)
        cost_adv_std = self.cost_advantages.view(self.num_envs*self.num_transitions_per_env, -1).std(0)
        
        # Normalized cost related
        self.cost_advantages = (self.cost_advantages - cost_adv_mean.view(1, 1, -1)) / (cost_adv_std.view(1, 1, -1) + 1e-8)
        
        # Cost violation
        # ((1 - gamma) * (returns - limit) + advantage_mean) / advantage_std
        # Use active thresholds for adaptive constraint scheduling
        self.cost_violation = ((1. - gamma) * (self.cost_returns - self.active_cost_d_values) + cost_adv_mean.view(1, 1, -1)) / (cost_adv_std.view(1, 1, -1) + 1e-8)

    def update_active_d_values(self, new_active_d):
        """
        Update active constraint thresholds for adaptive scheduling.
        
        Args:
            new_active_d: New threshold values, shape (num_costs,)
        """
        if self.active_cost_d_values is not None:
            self.active_cost_d_values.copy_(new_active_d)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        T_indices = (indices // self.num_envs).to(torch.long)
        B_indices = (indices % self.num_envs).to(torch.long)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                T_idx = T_indices[start:end]
                B_idx = B_indices[start:end]

                minibatch = self.get_minibatch_from_selection(T_idx, B_idx)
                
                # Expand standard minibatch with cost data
                target_cost_values_batch = self.cost_values[T_idx, B_idx]
                cost_advantages_batch = self.cost_advantages[T_idx, B_idx]
                cost_returns_batch = self.cost_returns[T_idx, B_idx]
                cost_violation_batch = self.cost_violation[T_idx, B_idx]

                yield minibatch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # We need to reimplement this partly or delegate to parent carefully
        # Parent class `recurrent_mini_batch_generator` yields standard minibatches
        # We need to augment them with cost data.
        
        # First call parent logic helpers to prepare data? 
        # Actually parent `recurrent_mini_batch_generator` is a generator itself, so we can't easily intercept inside.
        # We perform similar setup as parent.
        
        self._padded_obs_trajectories, self._trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.critic_observations is not None:
            self._padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_observations, self.dones)
            
        # Also need to split and pad cost/value trajectories? 
        # The parent `get_minibatch_from_selection` handles `padded_B_slice` for obs/critic/actions/etc.
        # But it won't handle our new cost buffers. 
        # So we probably need to pad our cost buffers too if we want to support recurrent cost learning efficiently.
        # However, checking NP3O implementation, it seems to assume non-recurrent or simple PPO logic usually.
        # But InstinctRL supports recurrent.
        # Let's see how `split_and_pad_trajectories` works. It takes (time, batch, ...) and done.
        
        # For cost buffers:
        self._padded_cost_values, _ = split_and_pad_trajectories(self.cost_values, self.dones)
        self._padded_cost_advantages, _ = split_and_pad_trajectories(self.cost_advantages, self.dones)
        self._padded_cost_returns, _ = split_and_pad_trajectories(self.cost_returns, self.dones)
        self._padded_cost_violation, _ = split_and_pad_trajectories(self.cost_violation, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Get standard minibatch using parent's method
                minibatch = self.get_minibatch_from_selection(
                    slice(None),
                    slice(start, stop),
                    padded_B_slice=slice(first_traj, last_traj),
                    prev_done_mask=last_was_done,
                )
                
                # Now extract cost data using the padded buffers
                # We need to mimic what get_minibatch_from_selection does for padded data
                # It seems it selects [T_select, padded_B_slice] which is [:, first_traj:last_traj]
                
                target_cost_values_batch = self._padded_cost_values[:, first_traj:last_traj]
                cost_advantages_batch = self._padded_cost_advantages[:, first_traj:last_traj]
                cost_returns_batch = self._padded_cost_returns[:, first_traj:last_traj]
                cost_violation_batch = self._padded_cost_violation[:, first_traj:last_traj]

                yield minibatch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch

                first_traj = last_traj
