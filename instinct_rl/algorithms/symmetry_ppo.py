from __future__ import annotations

import torch

from .ppo import PPO


class SymmetryPPO(PPO):
    """PPO variant that stores both original and mirrored feedforward transitions."""

    def act(
        self,
        obs,
        critic_obs,
        mirrored_obs: torch.Tensor | None = None,
        mirrored_critic_obs: torch.Tensor | None = None,
    ):
        if mirrored_obs is None:
            return super().act(obs, critic_obs)
        if self.actor_critic.is_recurrent:
            raise ValueError("SymmetryPPO currently supports feedforward actor-critic models only.")

        original_actions = self.actor_critic.act(obs).detach()
        original_values = self.actor_critic.evaluate(critic_obs if critic_obs is not None else obs).detach()
        original_log_prob = self.actor_critic.get_actions_log_prob(original_actions).detach()
        original_mu = self.actor_critic.action_mean.detach()
        original_sigma = self.actor_critic.action_std.detach()

        mirrored_actions = self.actor_critic.act(mirrored_obs).detach()
        mirrored_values = self.actor_critic.evaluate(
            mirrored_critic_obs if mirrored_critic_obs is not None else mirrored_obs
        ).detach()
        mirrored_log_prob = self.actor_critic.get_actions_log_prob(mirrored_actions).detach()
        mirrored_mu = self.actor_critic.action_mean.detach()
        mirrored_sigma = self.actor_critic.action_std.detach()

        self.transition.actions = torch.cat([original_actions, mirrored_actions], dim=0)
        self.transition.values = torch.cat([original_values, mirrored_values], dim=0)
        self.transition.actions_log_prob = torch.cat([original_log_prob, mirrored_log_prob], dim=0)
        self.transition.action_mean = torch.cat([original_mu, mirrored_mu], dim=0)
        self.transition.action_sigma = torch.cat([original_sigma, mirrored_sigma], dim=0)
        self.transition.observations = torch.cat([obs, mirrored_obs], dim=0)
        if critic_obs is not None or mirrored_critic_obs is not None:
            self.transition.critic_observations = torch.cat([critic_obs, mirrored_critic_obs], dim=0)
        else:
            self.transition.critic_observations = None
        return original_actions

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        next_obs,
        next_critic_obs,
        next_critic_obs_for_bootstrap=None,
        augmented_rewards: torch.Tensor | None = None,
        augmented_dones: torch.Tensor | None = None,
    ):
        self.transition.rewards = augmented_rewards.clone() if augmented_rewards is not None else rewards.clone()

        auxiliary_rewards = self.compute_auxiliary_reward(infos["observations"])
        for k, v in auxiliary_rewards.items():
            coef = getattr(self, f"{k}_coef", 1.0)
            if coef != 0.0:
                self.transition.rewards += coef * v * self.auxiliary_reward_per_env_reward_coefs
            infos["step"][k] = v

        self.transition.dones = augmented_dones if augmented_dones is not None else dones

        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        if "time_outs" in infos and bootstrap_obs is not None:
            with torch.no_grad():
                bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
            self.transition.rewards += (
                self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
