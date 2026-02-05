
from collections import namedtuple, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from instinct_rl.algorithms.np3o import NP3O
from instinct_rl.storage.rollout_storage_with_cost import RolloutStorageWithCost
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class RolloutStorageDreamWaQWithCost(RolloutStorageWithCost):
    """
    Storage class for DreamWaQ-NP3O, combining cost tracking with DreamWaQ's specific needs.
    """
    class Transition(RolloutStorageWithCost.Transition):
        def __init__(self):
            super().__init__()
            self.single_obs = None
            self.rewards_noClip = None

    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorageWithCost.MiniBatch._fields,
            "single_obs",
        ],
    )

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape,
                 num_single_obs, cost_shape=None, cost_d_values=None, num_rewards=1, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape,
                         cost_shape, cost_d_values, device=device)

        self.single_obs = torch.zeros(
            num_transitions_per_env, num_envs, num_single_obs, device=self.device
        )
        self.rewards_noClip = torch.zeros(
            num_transitions_per_env, num_envs, num_rewards, device=self.device
        )

    def add_transitions(self, transition: Transition):
        self.single_obs[self.step].copy_(transition.single_obs)
        self.rewards_noClip[self.step].copy_(transition.rewards_noClip.view(-1, self.num_rewards))
        super().add_transitions(transition)

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)

        # Handle single_obs (currently assuming MLP/non-recurrent for estimator part essentially)
        # Note: If CENet becomes recurrent, this needs more logic similar to hidden states
        if padded_B_slice is None:
            single_obs_batch = self.single_obs[T_select, B_select]
        else:
            single_obs_batch = self.single_obs[T_select, B_select]

        return RolloutStorageDreamWaQWithCost.MiniBatch(*minibatch, single_obs_batch)


class DreamWaQNP3O(NP3O):
    """
    DreamWaQ-NP3O: Combined Constrained RL (NP3O) with DreamWaQ Estimator.
    
    Inherits from NP3O to keep all constrained optimization logic, 
    and adds CENet integration from DreamWaQ.
    """
    def __init__(
        self,
        actor_critic,
        cenet,
        # DreamWaQ specific args
        vae_beta=0.5,
        use_Adaboot=True,
        # NP3O args (passed to super)
        k_value=1.0,
        k_value_max=1.0,
        k_value_growth_rate=1.0004,
        k_warmup_iterations=100,
        adaptive_alpha=0.02,
        cost_value_loss_coef=1.0,
        cost_viol_loss_coef=1.0,
        # PPO/Common args
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device='cpu',
        **kwargs
    ):
        super().__init__(
            actor_critic,
            k_value=k_value,
            k_value_max=k_value_max,
            k_value_growth_rate=k_value_growth_rate,
            k_warmup_iterations=k_warmup_iterations,
            adaptive_alpha=adaptive_alpha,
            cost_value_loss_coef=cost_value_loss_coef,
            cost_viol_loss_coef=cost_viol_loss_coef,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            learning_rate=learning_rate,
            device=device,
            **kwargs
        )
        
        # DreamWaQ components
        self.cenet = cenet
        self.cenet.to(self.device)
        self.vae_beta = vae_beta
        self.use_Adaboot = use_Adaboot
        
        self.cenet_loss_list = [torch.tensor(0.0, device=self.device) for _ in range(5)]
        self.Pboot = torch.tensor(1.0, device=self.device)
        self.optimizer_cenet = optim.Adam(self.cenet.parameters(), lr=learning_rate)
        self.use_estimate = False
        
        # Override transition with the custom one
        self.transition = RolloutStorageDreamWaQWithCost.Transition()

    def init_storage(
        self, 
        num_envs, 
        num_transitions_per_env, 
        obs_format, 
        num_actions, 
        num_rewards=1, 
        cost_shape=None, 
        cost_d_values=None,
        num_single_obs=0
    ):
        """
        Initialize rollout storage with both cost and DreamWaQ tracking.
        """
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for DreamWaQNP3O")
        
        # Initialize k_value tensor if needed
        if not isinstance(self.k_value, torch.Tensor):
            self.k_value = torch.tensor(
                [self.k_value] * cost_shape[0] if isinstance(cost_shape, tuple) else [self.k_value],
                dtype=torch.float32,
                device=self.device
            )
        else:
            self.k_value = self.k_value.to(self.device)
            
        self.storage = RolloutStorageDreamWaQWithCost(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_single_obs,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            num_rewards=num_rewards,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        """
        Sample action with CENet estimation and record cost values.
        """
        # --- DreamWaQ Logic: Estimate and Augment ---
        # 1. Encode observation to get latent struct
        estimate = self.cenet.encode(obs).detach()
        
        # 2. Determine final estimate based on usage flag (Adaboot)
        if self.use_estimate:
            final_estimate = estimate
        else:
            # If not using estimate, use ground truth for velocity (cheating) + zero latent
            # Note: This assumes ActorCritic knows how to handle segments
            # We need to construct the estimate manually: [lin_vel (3), zeros(latent)]
            lin_vel = get_subobs_by_components(
                critic_obs, 
                ["base_lin_vel"], 
                self.actor_critic.critic_obs_segments
            )
            # estimate is [batch, 19] typically (3 vel + 16 latent)
            # We take the latent part size from cenet config or infer from estimate shape
            latent_size = estimate.shape[-1] - 3 
            z_part = estimate[:, 3:] # Use learned latent or zeros? DreamWaQ uses learned latent part
            
            # Reconstruct: GT Vel + Latent
            final_estimate = torch.cat([lin_vel.detach(), z_part], dim=-1)
            
        # 3. Augment observation
        obs_augmented = torch.cat((obs, final_estimate), dim=-1)
        
        # --- Standard PPO/NP3O Act Logic ---
        # Note: We call actor_critic.act with augmented obs
        
        # We process recurrent states manually if needed, but actor_critic.act handles it usually
        # However, we need to populate self.transition fields correctly
        
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Action sampling
        self.transition.actions = self.actor_critic.act(obs_augmented).detach()
        
        # Value evaluation (Reward Critic)
        # Use critic_obs if available, else augmented obs (?) 
        # Usually critic uses privileged info directly.
        val_input = critic_obs if critic_obs is not None else obs_augmented
        self.transition.values = self.actor_critic.evaluate(val_input).detach()
        
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Record observations
        self.transition.observations = obs_augmented
        self.transition.critic_observations = critic_obs
        
        # --- NP3O Logic: Cost Evaluation ---
        with torch.no_grad():
            self.transition.cost_values = self.actor_critic.evaluate_cost(val_input).detach()
            
        return self.transition.actions

    def process_env_step(
        self, 
        rewards, 
        costs,
        dones, 
        infos, 
        rewards_noClip,
        num_single_obs,
        next_obs, 
        next_critic_obs, 
        next_critic_obs_for_bootstrap=None
    ):
        """
        Process step with support for Costs (NP3O) and SingleObs/RewardsNoClip (DreamWaQ).
        """
        # --- DreamWaQ Logic: Process "True" Next Obs for Estimator ---
        true_next_critic_obs = next_critic_obs.clone()
        if "termination_observations" in infos:
            term_ids = infos["termination_env_ids"]
            if len(term_ids) > 0:
                term_obs = infos["termination_observations"]
                if isinstance(term_obs, dict):
                    if "critic" in term_obs:
                        true_next_critic_obs[term_ids] = term_obs["critic"]
                else:
                    true_next_critic_obs[term_ids] = term_obs
        
        self.transition.single_obs = true_next_critic_obs[:,:num_single_obs]
        self.transition.rewards_noClip = rewards_noClip.clone()
        
        # --- NP3O Logic: Costs and Bootstrapping ---
        self.transition.costs = costs.clone()
        
        # Bootstrap Cost Values on Timeout
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        
        if 'time_outs' in infos and bootstrap_obs is not None:
            time_outs = infos['time_outs'].to(self.device)
            with torch.no_grad():
                # Cost Bootstrap
                bootstrap_cost_values = self.actor_critic.evaluate_cost(bootstrap_obs).detach()
                self.transition.costs = self.transition.costs + self.gamma * bootstrap_cost_values * time_outs.unsqueeze(-1)
                
                # Reward Value Bootstrap
                bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
                # Assuming rewards (vectorized) + gamma * ...
                # Note: NP3O's parent (PPO) process_env_step calls super().process_env_step
                # We need to manually handle reward bootstrap since we are overriding
        
        # Standard PPO process (rewards, values, dones) - partially handled by parent call in NP3O
        # But we need to pass extra args to NP3O.process_env_step which calls PPO.process_env_step...
        # BUT NP3O's process_env_step signature is fixed. 
        # We are overriding it here completely to mix logic.
        
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Reward Value Bootstrap (Manual implementation to ensure correctness with mixed logic)
        if 'time_outs' in infos and bootstrap_obs is not None:
             with torch.no_grad():
                bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
             self.transition.rewards += (
                self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)
            )

        # Add to storage
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_Pboot(self):
        """Adaptive Bootstrapping logic from DreamWaQ."""
        update_cenet = True
        if self.use_Adaboot:
            rewards_noClip = self.storage.rewards_noClip.clone()
            # Sum over time steps, resulting in (num_envs,)
            rewards_sum = rewards_noClip.sum(dim=0).squeeze(-1) 
            
            mean_episodic_rewards = torch.mean(rewards_sum)
            std_episodic_rewards = torch.std(rewards_sum)

            # CV
            cv = std_episodic_rewards / mean_episodic_rewards if mean_episodic_rewards > 0 else torch.tensor(10086.0).to(self.device)

            self.Pboot = 1 - torch.tanh(cv)
            if self.Pboot > torch.rand(1, dtype=torch.float32, device=self.device):
                update_cenet = True
            else:
                update_cenet = False
        return update_cenet

    def update(self, current_learning_iteration):
        """
        Combined Update Loop:
        1. Decide Pboot status.
        2. Iterate minibatches.
        3. Compute NP3O losses (PPO + Cost).
        4. Compute CENet losses (Reconstruction + KL).
        5. Update both networks.
        """
        self.current_learning_iteration = current_learning_iteration
        
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        
        # 1. Compute Pboot
        self.use_estimate = self.compute_Pboot()
        
        # Generator
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # CENet Loss Trackers
        mean_loss_vt = torch.tensor(0.0, device=self.device)
        mean_loss_ot = torch.tensor(0.0, device=self.device)
        mean_loss_kl = torch.tensor(0.0, device=self.device)
        mean_loss_est = torch.tensor(0.0, device=self.device)
        
        for minibatch_data in generator:
            # Unpack (RolloutStorageDreamWaQWithCost.MiniBatch)
            # fields: [obs, critic_obs, actions, ... , single_obs]
            minibatch = minibatch_data[0]
            target_cost_values_batch = minibatch_data[1]
            cost_advantages_batch = minibatch_data[2]
            cost_returns_batch = minibatch_data[3]
            cost_violation_batch = minibatch_data[4]
            single_obs_batch = minibatch.single_obs

            # --- Step A: NP3O (Actor-Critic) Update ---
            
            # A1. Standard PPO losses
            losses, _, stats = self.compute_losses(minibatch)
            
            # A2. Cost Critic Loss
            critic_hidden_states = minibatch.hidden_states.critic if self.actor_critic.is_recurrent else None
            cost_value_batch = self.actor_critic.evaluate_cost(
                minibatch.critic_obs, 
                masks=minibatch.masks, 
                hidden_states=critic_hidden_states
            )
            
            if self.use_clipped_value_loss:
                cost_value_clipped = target_cost_values_batch + (
                    cost_value_batch - target_cost_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            # A3. Violation Loss
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
            viol_loss = self.compute_viol(
                actions_log_prob_batch, 
                minibatch.old_actions_log_prob, 
                cost_advantages_batch, 
                cost_violation_batch
            )

            # A4. Combine Policy/Critic Losses
            total_ac_loss = 0.0
            for k, v in losses.items():
                coef = getattr(self, k + "_coef", 1.0)
                total_ac_loss = total_ac_loss + coef * v
                mean_losses[k] += v.detach()
            
            total_ac_loss = total_ac_loss + self.cost_value_loss_coef * cost_value_loss
            total_ac_loss = total_ac_loss + self.cost_viol_loss_coef * viol_loss
            
            mean_losses["cost_value_loss"] += cost_value_loss.detach()
            mean_losses["viol_loss"] += viol_loss.detach()
            mean_losses["total_ac_loss"] += total_ac_loss.detach()

            # A5. Gradient Step (Actor-Critic) - Manual like DreamWaQ
            self.optimizer.zero_grad()
            total_ac_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Collect AC stats
            for k, v in stats.items():
                if k not in average_stats: average_stats[k] = 0.0
                average_stats[k] += v.detach()

            # --- Step B: CENet (Estimator) Update ---
            
            # B1. Get Ground Truth Velocity
            # Requires critic_obs_segments to be available in actor_critic or passed somehow
            # DreamWaQ assumes actor_critic has it.
            lin_vel = get_subobs_by_components(
                minibatch.critic_obs, 
                ["base_lin_vel"], 
                self.actor_critic.critic_obs_segments
            )
            
            # B2. Strip Estimate from Obs to get Raw Obs
            # obs is [raw | estimate]. CENet takes raw.
            # raw input dim is in self.cenet.encoder.model[0].in_features
            raw_obs_dim = self.cenet.encoder.model[0].in_features
            raw_obs = minibatch.obs[..., :raw_obs_dim]
            
            # B3. Forward CENet
            z_sample = self.cenet.encode(raw_obs)
            est_mean = self.cenet.encoder_mean
            est_logvar = self.cenet.encoder_logvar
            
            # B4. KL Loss
            # est_logvar is 2*log_std = log(var)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + est_logvar - est_mean[:,-16:] ** 2 - torch.exp(est_logvar), dim=1))
            
            # B5. Functional Losses (Velocity Tracking + Observation Recruitment)
            loss_vt = F.mse_loss(est_mean[:,:3], lin_vel)
            loss_ot = F.mse_loss(self.cenet.decode(z_sample), single_obs_batch)
            
            cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss
            
            # B6. Gradient Step (CENet)
            self.optimizer_cenet.zero_grad()
            cenet_loss.backward()
            nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
            self.optimizer_cenet.step()
            
            mean_loss_vt += loss_vt.detach()
            mean_loss_ot += loss_ot.detach()
            mean_loss_kl += kl_loss.detach()
            mean_loss_est += cenet_loss.detach()

        # --- Statistics & Cleanup ---
        num_updates = self.num_learning_epochs * self.num_mini_batches
        
        # Normalize losses
        for k in mean_losses:
            mean_losses[k] /= num_updates
        for k in average_stats:
            average_stats[k] /= num_updates
            
        mean_loss_vt /= num_updates
        mean_loss_ot /= num_updates
        mean_loss_kl /= num_updates
        mean_loss_est /= num_updates
        
        self.cenet_loss_list = [mean_loss_vt, mean_loss_ot, mean_loss_kl, mean_loss_est, self.Pboot]
        
        # Clear storage
        self.storage.clear()
        
        # Clip AC std
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min=self.clip_min_std)
            
        # Update NP3O K-value
        self.update_k_value()
        
        # Add extra stats
        average_stats.update({
            "k_value": self.k_value.mean() if isinstance(self.k_value, torch.Tensor) else self.k_value,
            "estimator_loss": mean_loss_est,
            "estimator_mse_vt": mean_loss_vt,
            "estimator_mse_ot": mean_loss_ot,
            "estimator_kl": mean_loss_kl,
            "Pboot": self.Pboot
        })
        
        # Merge estimator losses into mean_losses for logging
        mean_losses["estimator_loss"] = mean_loss_est

        return mean_losses, average_stats
