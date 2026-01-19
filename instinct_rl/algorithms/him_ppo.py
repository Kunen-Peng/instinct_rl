import torch

from instinct_rl.algorithms.ppo import PPO


class HIMPPO(PPO):
    """Hierarchical Imitation Mode PPO - optimized for ObservationManager history.
    
    This algorithm extends standard PPO by:
    1. Supporting observation history directly from ObservationManager (via CircularBuffer)
    2. Accepting next_critic_obs_for_bootstrap parameter for accurate value estimation
    3. Ensuring correct observation ordering (oldest_first format from CircularBuffer)
    
    Key optimization: Instead of storing history separately, this version leverages
    the fact that ObservationManager already maintains observation history via CircularBuffer.
    The policy observations are already flattened history tensors when returned from
    compute_group() with flatten_history_dim=True.
    
    Observation format expected:
    - Policy observations: [obs_t0, obs_t1, ..., obs_t(H-1)] (oldest_first)
    - Shape: [batch_size, history_size * num_one_step_obs]
    """

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs, next_critic_obs_for_bootstrap=None):
        """Process environment step with termination observation bootstrapping support.
        
        Optimized for ObservationManager's CircularBuffer history output.
        
        The next_obs parameter already contains the full observation history maintained
        by ObservationManager's CircularBuffer in oldest_first format:
        [obs_t0, obs_t1, ..., obs_t(H-1)]
        
        Args:
            rewards: Rewards from the environment
            dones: Done flags
            infos: Additional information dict (includes observations group)
            next_obs: Next policy observations (already contains full history from CircularBuffer)
            next_critic_obs: Next critic observations (current step)
            next_critic_obs_for_bootstrap: Next critic observations with termination obs (optional)
                If provided, this will be used for value bootstrapping on timeouts
        """
        self.transition.rewards = rewards.clone()

        auxiliary_rewards = self.compute_auxiliary_reward(infos["observations"])
        # Add auxiliary rewards to the transition
        for k, v in auxiliary_rewards.items():
            coef = getattr(
                self, f"{k}_coef", 1.0
            )
            if coef != 0.0:
                self.transition.rewards += coef * v * self.auxiliary_reward_per_env_reward_coefs
            infos["step"][k] = v

        self.transition.dones = dones
        
        # Bootstrapping on time outs using bootstrap observations
        # Use next_critic_obs_for_bootstrap if provided (contains termination obs)
        # Otherwise use regular next_critic_obs
        if "time_outs" in infos:
            bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
            if bootstrap_obs is not None:
                # Compute values for next state to bootstrap truncated episodes
                with torch.no_grad():
                    bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
                self.transition.rewards += (
                    self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)
                )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute returns using the last critic observation.
        
        This method is used to compute returns at the end of a rollout phase.
        Compatible with both standard and HIM modes.
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
