import torch
import torch.nn as nn
from instinct_rl.algorithms.ppo import PPO


class HIMPPO(PPO):
    """Hierarchical Imitation Mode PPO.
    
    Reference: HIMLoco/rsl_rl/algorithms/him_ppo.py
    
    Key difference from standard PPO: Uses next_critic_obs for estimator training.
    """

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards):
        """Initialize storage with next_critic_obs support."""
        # Import here to avoid circular dependency
        from instinct_rl.storage.rollout_storage import SarsaRolloutStorage
        from instinct_rl.utils.utils import get_subobs_size
        
        obs_shape = (get_subobs_size(obs_format["policy"]),)
        critic_obs_shape = (get_subobs_size(obs_format.get("critic", obs_format["policy"])),)
        actions_shape = (num_actions,)
        
        # Use SarsaRolloutStorage which supports next_observations
        self.storage = SarsaRolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            num_rewards=num_rewards,
            device=self.device,
        )
        
        # Create transition object from storage's Transition class
        self.transition = self.storage.Transition()

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs, next_critic_obs_for_bootstrap=None):
        """Process environment step.
        
        Args:
            rewards: Environment rewards
            dones: Done flags
            infos: Info dict
            next_obs: Next policy observations
            next_critic_obs: Next critic observations
            next_critic_obs_for_bootstrap: Bootstrap observations (with termination obs if available)
        """
        self.transition.rewards = rewards.clone()
        
        # Add auxiliary rewards if configured
        if hasattr(self, "compute_auxiliary_reward"):
            auxiliary_rewards = self.compute_auxiliary_reward(infos.get("observations", {}))
            for k, v in auxiliary_rewards.items():
                coef = getattr(self, f"{k}_coef", 1.0)
                if coef != 0.0:
                    self.transition.rewards += coef * v * getattr(self, "auxiliary_reward_per_env_reward_coefs", 1.0)
                if "step" in infos:
                    infos["step"][k] = v

        self.transition.dones = dones
        
        # Store next observations (required by SarsaRolloutStorage)
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
            if bootstrap_obs is not None:
                with torch.no_grad():
                    bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
                self.transition.rewards += (
                    self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)
                )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute returns using last critic observation."""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, current_learning_iteration):
        """Update policy and estimator."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        
        # Prepare for update
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_updates = 0
        for batch in generator:
            # Unpack batch from SarsaRolloutStorage.MiniBatch
            # MiniBatch includes: obs, critic_obs, actions, values, advantages, returns, 
            #                      old_actions_log_prob, old_mu, old_sigma, hidden_states, masks,
            #                      next_obs, next_critic_obs
            obs_batch = batch.obs
            critic_obs_batch = batch.critic_obs
            actions_batch = batch.actions
            target_values_batch = batch.values
            advantages_batch = batch.advantages
            returns_batch = batch.returns
            old_actions_log_prob_batch = batch.old_actions_log_prob
            old_mu_batch = batch.old_mu
            old_sigma_batch = batch.old_sigma
            next_critic_obs_batch = batch.next_critic_obs  # This is the key for estimator!

            # Forward pass
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL Adaptation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                        (2.0 * torch.square(sigma_batch)) - 0.5, 
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Estimator Update (uses layout conversion if needed)
            estimation_loss, swap_loss = self.actor_critic.update_estimator(
                obs_batch, next_critic_obs_batch, lr=self.learning_rate
            )

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            num_updates += 1

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.storage.clear()

        # Return as tensors for compatibility with on_policy_runner.gather_stat_values
        return {
            "value_loss": torch.tensor(mean_value_loss, device=self.device),
            "surrogate_loss": torch.tensor(mean_surrogate_loss, device=self.device),
            "estimation_loss": torch.tensor(mean_estimation_loss, device=self.device),
            "swap_loss": torch.tensor(mean_swap_loss, device=self.device),
        }, {}

