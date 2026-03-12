import math
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from instinct_rl.algorithms.np3o import NP3O
from instinct_rl.storage.rollout_storage_with_cost import RolloutStorageWithCost
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class RolloutStorageDreamWaQWithCost(RolloutStorageWithCost):
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

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        critic_obs_shape,
        actions_shape,
        num_single_obs,
        cost_shape=None,
        cost_d_values=None,
        num_rewards=1,
        device="cpu",
    ):
        super().__init__(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            cost_shape,
            cost_d_values,
            device=device,
        )
        self.single_obs = torch.zeros(num_transitions_per_env, num_envs, num_single_obs, device=self.device)
        self.rewards_noClip = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)

    def add_transitions(self, transition: Transition):
        self.single_obs[self.step].copy_(transition.single_obs)
        self.rewards_noClip[self.step].copy_(transition.rewards_noClip.view(-1, self.num_rewards))
        super().add_transitions(transition)

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
        single_obs_batch = self.single_obs[T_select, B_select]
        return RolloutStorageDreamWaQWithCost.MiniBatch(*minibatch, single_obs_batch)


class RolloutStorageDreamWaQWithCostRecurrent(RolloutStorageDreamWaQWithCost):
    class Transition(RolloutStorageDreamWaQWithCost.Transition):
        def __init__(self):
            super().__init__()
            self.cenet_hidden_states = None

    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorageDreamWaQWithCost.MiniBatch._fields,
            "cenet_hidden_states",
        ],
    )

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        critic_obs_shape,
        actions_shape,
        num_single_obs,
        cenet_hidden_state_shape,
        cost_shape=None,
        cost_d_values=None,
        num_rewards=1,
        device="cpu",
    ):
        super().__init__(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            num_single_obs,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            num_rewards=num_rewards,
            device=device,
        )
        self.cenet_hidden_states = torch.zeros(
            num_transitions_per_env, num_envs, *cenet_hidden_state_shape, device=self.device
        )

    def add_transitions(self, transition: Transition):
        if transition.cenet_hidden_states is not None:
            self.cenet_hidden_states[self.step].copy_(transition.cenet_hidden_states)
        super().add_transitions(transition)

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
        cenet_hidden_states_batch = self.cenet_hidden_states[T_select, B_select]
        return RolloutStorageDreamWaQWithCostRecurrent.MiniBatch(*minibatch, cenet_hidden_states_batch)


class DreamWaQNP3OCommon(NP3O):
    def __init__(
        self,
        actor_critic,
        cenet,
        vae_beta=0.5,
        use_Adaboot=True,
        num_estimator_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        **kwargs,
    ):
        super().__init__(actor_critic, learning_rate=learning_rate, device=device, **kwargs)
        self.cenet = cenet
        self.cenet.to(self.device)
        self.vae_beta = vae_beta
        self.use_Adaboot = use_Adaboot
        self.num_estimator_epochs = num_estimator_epochs
        self.target_mse_ot = kwargs.pop("target_mse_ot", 0.05)
        self.beta_delta = kwargs.pop("beta_delta", 1.0)
        self.min_vae_beta = kwargs.pop("min_vae_beta", 0.01)
        self.max_vae_beta = kwargs.pop("max_vae_beta", 0.5)
        self.cenet_loss_list = [torch.tensor(0.0, device=self.device) for _ in range(5)]
        self.Pboot = torch.tensor(1.0, device=self.device)
        self.optimizer_cenet = optim.Adam(self.cenet.parameters(), lr=learning_rate)
        self.use_estimate = False
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
        num_single_obs=0,
    ):
        self._policy_obs_segments = obs_format["policy"]
        self._critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for DreamWaQNP3O")
        self.k_value = self._to_cost_tensor(self.k_value, cost_shape)
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
        estimate = self.cenet.encode(obs).detach()
        if self.use_estimate:
            final_estimate = estimate
        else:
            lin_vel = get_subobs_by_components(critic_obs, ["base_lin_vel"], self._critic_obs_segments)
            z_part = estimate[:, 3:]
            final_estimate = torch.cat([lin_vel.detach(), z_part], dim=-1)
        obs_augmented = torch.cat((obs, final_estimate), dim=-1)

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs_augmented).detach()
        val_input = critic_obs if critic_obs is not None else obs_augmented
        self.transition.values = self.actor_critic.evaluate(val_input).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs_augmented
        self.transition.critic_observations = critic_obs
        with torch.no_grad():
            self.transition.cost_values = self._evaluate_cost(val_input).detach()
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
        next_critic_obs_for_bootstrap=None,
    ):
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

        self.transition.single_obs = true_next_critic_obs[:, :num_single_obs]
        self.transition.rewards_noClip = rewards_noClip.clone()
        self.transition.costs = costs.clone()
        bootstrap_obs = next_critic_obs_for_bootstrap if next_critic_obs_for_bootstrap is not None else next_critic_obs
        self.transition.costs = self._bootstrap_timeout_costs(self.transition.costs, infos, bootstrap_obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos and bootstrap_obs is not None:
            with torch.no_grad():
                bootstrap_values = self.actor_critic.evaluate(bootstrap_obs).detach()
            self.transition.rewards += self.gamma * bootstrap_values * infos["time_outs"].unsqueeze(1).to(self.device)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_use_estimate(self):
        use_estimate = True
        if self.use_Adaboot:
            rewards_noClip = self.storage.rewards_noClip.clone()
            rewards_sum = rewards_noClip.sum(dim=0).squeeze(-1)
            mean_episodic_rewards = torch.mean(rewards_sum)
            std_episodic_rewards = torch.std(rewards_sum)
            cv = (
                std_episodic_rewards / mean_episodic_rewards
                if mean_episodic_rewards > 0
                else torch.tensor(10086.0).to(self.device)
            )
            self.Pboot = 1 - torch.tanh(cv)
            use_estimate = self.Pboot > torch.rand(1, dtype=torch.float32, device=self.device)
        return use_estimate

    def _init_estimator_stats(self):
        return {
            "loss_vt": torch.tensor(0.0, device=self.device),
            "loss_ot": torch.tensor(0.0, device=self.device),
            "loss_kl": torch.tensor(0.0, device=self.device),
            "loss_est": torch.tensor(0.0, device=self.device),
        }

    def _finalize_estimator_stats(self, estimator_stats, mean_losses, average_stats):
        num_updates = self.num_learning_epochs * self.num_mini_batches
        est_updates = num_updates * self.num_estimator_epochs

        self._average_accumulator(mean_losses, num_updates)
        self._average_accumulator(average_stats, num_updates)
        for key in estimator_stats.keys():
            estimator_stats[key] /= est_updates

        k_beta = math.exp(self.beta_delta * (self.target_mse_ot - estimator_stats["loss_ot"].item()))
        self.vae_beta = max(self.min_vae_beta, min(self.max_vae_beta, self.vae_beta * k_beta))
        self.cenet_loss_list = [
            estimator_stats["loss_vt"],
            estimator_stats["loss_ot"],
            estimator_stats["loss_kl"],
            estimator_stats["loss_est"],
            self.Pboot,
        ]

        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min=self.clip_min_std)
        self.update_k_value()

        average_stats.update(
            {
                "k_value": self.k_value.mean() if isinstance(self.k_value, torch.Tensor) else self.k_value,
                "estimator_loss": estimator_stats["loss_est"],
                "estimator_mse_vt": estimator_stats["loss_vt"],
                "estimator_mse_ot": estimator_stats["loss_ot"],
                "estimator_kl": estimator_stats["loss_kl"],
                "Pboot": self.Pboot,
            }
        )
        mean_losses["estimator_loss"] = estimator_stats["loss_est"]
        return mean_losses, average_stats


class DreamWaQNP3ORecurrentCommon(DreamWaQNP3OCommon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition = RolloutStorageDreamWaQWithCostRecurrent.Transition()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_format,
        num_actions,
        num_rewards=1,
        cost_shape=None,
        cost_d_values=None,
        num_single_obs=0,
    ):
        self._policy_obs_segments = obs_format["policy"]
        self._critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for DreamWaQNP3O")
        self.k_value = self._to_cost_tensor(self.k_value, cost_shape)

        cenet_hidden_state_shape = (self.cenet.rnn.num_layers, self.cenet.rnn.hidden_size)
        self.storage = RolloutStorageDreamWaQWithCostRecurrent(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_single_obs,
            cenet_hidden_state_shape,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            num_rewards=num_rewards,
            device=self.device,
        )

    def act(self, obs, critic_obs, cenet_hidden_states=None):
        if obs.dim() == 2:
            rnn_input = obs.unsqueeze(1)
        else:
            rnn_input = obs

        emb = self.cenet.pre_embedding(rnn_input)
        rnn_out, next_cenet_hidden_states = self.cenet.rnn(emb, cenet_hidden_states)
        last_features = rnn_out[:, -1, :]
        v_mean = self.cenet.v_head(last_features)
        z_mean = self.cenet.z_mean_head(last_features)
        z_log_std = self.cenet.z_logstd_head(last_features)
        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)
        z_std = torch.exp(z_log_std)
        z_sampled = Normal(z_mean, z_std).rsample()
        estimate = torch.cat([v_mean, z_sampled], dim=-1).detach()

        if self.use_estimate:
            final_estimate = estimate
        else:
            lin_vel = get_subobs_by_components(critic_obs, ["base_lin_vel"], self._critic_obs_segments)
            z_part = estimate[:, 3:]
            final_estimate = torch.cat([lin_vel.detach(), z_part], dim=-1)

        obs_augmented = torch.cat((obs, final_estimate), dim=-1)

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs_augmented).detach()
        val_input = critic_obs if critic_obs is not None else obs_augmented
        self.transition.values = self.actor_critic.evaluate(val_input).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs_augmented
        self.transition.critic_observations = critic_obs
        self.transition.cenet_hidden_states = cenet_hidden_states.permute(1, 0, 2)
        with torch.no_grad():
            self.transition.cost_values = self._evaluate_cost(val_input).detach()
        return self.transition.actions, next_cenet_hidden_states
