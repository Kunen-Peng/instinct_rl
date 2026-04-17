import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.dreamwaq_common import PPODreamWaQRecurrentCommon, RolloutStorageDreamWaQRecurrent
from instinct_rl.storage.rollout_storage import RolloutStorage
from instinct_rl.utils import split_and_pad_trajectories
from instinct_rl.utils.buffer import buffer_method
from instinct_rl.utils.utils import get_subobs_by_components


class RolloutStorageDreamWaQRecurrentV3(RolloutStorageDreamWaQRecurrent):
    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorageDreamWaQRecurrent.MiniBatch._fields,
            "aug_obs",
            "aug_critic_obs",
            "aug_actions",
            "aug_old_actions_log_prob",
            "aug_returns",
            "aug_advantages",
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
            cenet_hidden_state_shape,
            num_rewards=num_rewards,
            device=device,
        )
        self.clear_sgma_augmentation()

    def clear(self):
        super().clear()
        self.clear_sgma_augmentation()

    def clear_sgma_augmentation(self):
        self.aug_observations = None
        self.aug_critic_observations = None
        self.aug_actions = None
        self.aug_old_actions_log_prob = None
        self.aug_returns = None
        self.aug_advantages = None

    def set_sgma_augmentation(
        self,
        aug_observations,
        aug_critic_observations,
        aug_actions,
        aug_old_actions_log_prob,
        aug_returns,
        aug_advantages,
    ):
        self.aug_observations = aug_observations
        self.aug_critic_observations = aug_critic_observations
        self.aug_actions = aug_actions
        self.aug_old_actions_log_prob = aug_old_actions_log_prob
        self.aug_returns = aug_returns
        self.aug_advantages = aug_advantages

    def get_direct_policy_obs_sequences(self, raw_encoder_input_dim):
        return self.observations[..., :raw_encoder_input_dim]

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        self._padded_obs_trajectories, self._trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        self._padded_raw_obs_trajectories, _ = split_and_pad_trajectories(self.raw_observations, self.dones)
        if self.critic_observations is not None:
            self._padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_observations, self.dones)
        if self.raw_critic_observations is not None:
            self._padded_raw_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.raw_critic_observations, self.dones
            )
        self._padded_single_obs_trajectories, _ = split_and_pad_trajectories(self.single_obs, self.dones)
        self._padded_dones_trajectories, _ = split_and_pad_trajectories(self.dones.float(), self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
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

                yield self.get_minibatch_from_selection(
                    slice(None),
                    slice(start, stop),
                    padded_B_slice=slice(first_traj, last_traj),
                    prev_done_mask=last_was_done,
                )

                first_traj = last_traj

    def _get_padded_cenet_hidden_states(self, padded_B_slice, prev_done_mask):
        traj_start_mask = prev_done_mask.permute(1, 0)
        return (
            self.cenet_hidden_states.permute(1, 0, 2, 3)[traj_start_mask][padded_B_slice]
            .transpose(1, 0)
            .contiguous()
        )

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        if padded_B_slice is None:
            minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
            aug_obs_batch = self.aug_observations[T_select, B_select] if self.aug_observations is not None else None
            if self.aug_critic_observations is not None:
                aug_critic_obs_batch = self.aug_critic_observations[T_select, B_select]
            else:
                aug_critic_obs_batch = None
            aug_actions_batch = self.aug_actions[T_select, B_select] if self.aug_actions is not None else None
            aug_old_actions_log_prob_batch = (
                self.aug_old_actions_log_prob[T_select, B_select] if self.aug_old_actions_log_prob is not None else None
            )
            aug_returns_batch = self.aug_returns[T_select, B_select] if self.aug_returns is not None else None
            aug_advantages_batch = self.aug_advantages[T_select, B_select] if self.aug_advantages is not None else None
            return RolloutStorageDreamWaQRecurrentV3.MiniBatch(
                *minibatch,
                aug_obs_batch,
                aug_critic_obs_batch,
                aug_actions_batch,
                aug_old_actions_log_prob_batch,
                aug_returns_batch,
                aug_advantages_batch,
            )

        obs_batch = self._padded_obs_trajectories[T_select, padded_B_slice]
        raw_obs_batch = self._padded_raw_obs_trajectories[T_select, padded_B_slice]
        critic_obs_batch = (
            obs_batch
            if self.critic_observations is None
            else self._padded_critic_obs_trajectories[T_select, padded_B_slice]
        )
        if self.raw_critic_observations is None:
            raw_critic_obs_batch = raw_obs_batch
        else:
            raw_critic_obs_batch = self._padded_raw_critic_obs_trajectories[T_select, padded_B_slice]
        obs_mask_batch = self._trajectory_masks[T_select, padded_B_slice]
        single_obs_batch = self._padded_single_obs_trajectories[T_select, padded_B_slice]
        dones_batch = self._padded_dones_trajectories[T_select, padded_B_slice]
        cenet_hidden_states_batch = self._get_padded_cenet_hidden_states(padded_B_slice, prev_done_mask)

        hid_batch = None
        if self.saved_hidden_states is not None:
            traj_start_mask = prev_done_mask.permute(1, 0)
            hid_batch = buffer_method(
                buffer_method(
                    buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[traj_start_mask][padded_B_slice],
                    "transpose",
                    1,
                    0,
                ),
                "contiguous",
            )

        action_batch = self.actions[T_select, B_select]
        target_value_batch = self.values[T_select, B_select]
        return_batch = self.returns[T_select, B_select]
        old_action_log_prob_batch = self.actions_log_prob[T_select, B_select]
        advantage_batch = self.advantages[T_select, B_select]
        old_mu_batch = self.mu[T_select, B_select]
        old_sigma_batch = self.sigma[T_select, B_select]
        aug_obs_batch = self.aug_observations[T_select, B_select] if self.aug_observations is not None else None
        if self.aug_critic_observations is not None:
            aug_critic_obs_batch = self.aug_critic_observations[T_select, B_select]
        else:
            aug_critic_obs_batch = None
        aug_actions_batch = self.aug_actions[T_select, B_select] if self.aug_actions is not None else None
        aug_old_actions_log_prob_batch = (
            self.aug_old_actions_log_prob[T_select, B_select] if self.aug_old_actions_log_prob is not None else None
        )
        aug_returns_batch = self.aug_returns[T_select, B_select] if self.aug_returns is not None else None
        aug_advantages_batch = self.aug_advantages[T_select, B_select] if self.aug_advantages is not None else None

        return RolloutStorageDreamWaQRecurrentV3.MiniBatch(
            obs_batch,
            critic_obs_batch,
            raw_obs_batch,
            raw_critic_obs_batch,
            action_batch,
            target_value_batch,
            advantage_batch,
            return_batch,
            old_action_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_batch,
            obs_mask_batch,
            single_obs_batch,
            dones_batch,
            cenet_hidden_states_batch,
            aug_obs_batch,
            aug_critic_obs_batch,
            aug_actions_batch,
            aug_old_actions_log_prob_batch,
            aug_returns_batch,
            aug_advantages_batch,
        )


class PPODreamWaQRecurrentV3(PPODreamWaQRecurrentCommon):
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, num_single_obs=0):
        import numpy as np

        obs_size = sum(np.prod(v) for v in obs_format["policy"].values())
        critic_obs_size = (
            sum(np.prod(v) for v in obs_format["critic"].values()) if "critic" in obs_format else None
        )

        if self.cenet.rnn is not None:
            cenet_hidden_state_shape = (self.cenet.rnn.num_layers, self.cenet.rnn.hidden_size)
            self.storage = RolloutStorageDreamWaQRecurrentV3(
                num_envs,
                num_transitions_per_env,
                [obs_size],
                [critic_obs_size],
                [num_actions],
                num_single_obs,
                cenet_hidden_state_shape,
                num_rewards=num_rewards,
                device=self.device,
            )
        else:
            # MLP mode — delegate to base class (non-recurrent storage).
            super().init_storage(
                num_envs, num_transitions_per_env, obs_format, num_actions,
                num_rewards=num_rewards, num_single_obs=num_single_obs,
            )

    def _compute_recurrent_estimator_losses(self, minibatch):
        """Compute estimator losses over padded trajectory batches (RNN only)."""
        lin_vel = get_subobs_by_components(
            minibatch.critic_obs,
            ["base_lin_vel"],
            self.actor_critic.critic_obs_segments,
        )
        raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
        valid = minibatch.masks.bool()
        hidden_states = minibatch.cenet_hidden_states

        z_sample = self.cenet.encode(raw_obs, hidden_states=hidden_states, masks=minibatch.masks)
        est_mean = self.cenet.encoder_mean
        est_logvar = self.cenet.encoder_logvar
        est_obs = self.cenet.decode(z_sample)

        z_mean = est_mean[..., self.cenet.dim_v :]
        kl_terms = -0.5 * torch.sum(1 + est_logvar - z_mean.square() - torch.exp(est_logvar), dim=-1)
        kl_loss = kl_terms[valid].mean()
        loss_vt = F.mse_loss(est_mean[..., : self.cenet.dim_v][valid], lin_vel[valid])
        loss_ot = self.compute_weighted_ot_loss(est_obs, minibatch.single_obs, valid)
        return loss_vt, loss_ot, kl_loss

    def _compute_mlp_estimator_losses(self, minibatch):
        """Compute estimator losses over flat mini-batches (MLP only)."""
        lin_vel = get_subobs_by_components(
            minibatch.critic_obs,
            ["base_lin_vel"],
            self.actor_critic.critic_obs_segments,
        )
        raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
        valid = ~minibatch.dones.squeeze(-1).bool()

        z_sample = self.cenet.encode(raw_obs)
        est_mean = self.cenet.encoder_mean
        est_logvar = self.cenet.encoder_logvar
        est_obs = self.cenet.decode(z_sample)

        z_mean = est_mean[..., self.cenet.dim_v :]
        kl_terms = -0.5 * torch.sum(1 + est_logvar - z_mean.square() - torch.exp(est_logvar), dim=-1)
        kl_loss = kl_terms[valid].mean()
        loss_vt = F.mse_loss(est_mean[..., : self.cenet.dim_v][valid], lin_vel[valid])
        loss_ot = self.compute_weighted_ot_loss(est_obs, minibatch.single_obs, valid)
        return loss_vt, loss_ot, kl_loss

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        accumulators = self._init_update_stats()
        self.use_estimate = self.compute_use_estimate()

        ppo_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in ppo_generator:
            losses, _, _ = self.compute_losses(minibatch)
            loss = (
                losses["surrogate_loss"]
                + self.value_loss_coef * losses["value_loss"]
                + losses.get("entropy", 0.0) * self.entropy_coef
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            accumulators["value_loss"] += losses["value_loss"]
            accumulators["surrogate_loss"] += losses["surrogate_loss"]

        if self.cenet.rnn is not None:
            # RNN mode: use recurrent mini-batch generator with padded trajectories.
            estimator_generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for minibatch in estimator_generator:
                for _ in range(self.num_estimator_epochs):
                    loss_vt, loss_ot, kl_loss = self._compute_recurrent_estimator_losses(minibatch)
                    cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss

                    self.optimizer_cenet.zero_grad()
                    cenet_loss.backward()
                    nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
                    self.optimizer_cenet.step()

                    accumulators["loss_vt"] += loss_vt
                    accumulators["loss_ot"] += loss_ot
                    accumulators["loss_kl"] += kl_loss
                    accumulators["loss_est"] += cenet_loss
        else:
            # MLP mode: use standard mini-batch generator.
            est_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for minibatch in est_generator:
                for _ in range(self.num_estimator_epochs):
                    loss_vt, loss_ot, kl_loss = self._compute_mlp_estimator_losses(minibatch)
                    cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss

                    self.optimizer_cenet.zero_grad()
                    cenet_loss.backward()
                    nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
                    self.optimizer_cenet.step()

                    accumulators["loss_vt"] += loss_vt
                    accumulators["loss_ot"] += loss_ot
                    accumulators["loss_kl"] += kl_loss
                    accumulators["loss_est"] += cenet_loss

        return self._finalize_update(accumulators)


class PPODreamWaQRecurrentV3SGMA(PPODreamWaQRecurrentV3):
    def __init__(self, *args, symmetry_helper_class_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if symmetry_helper_class_name is None:
            raise ValueError("PPODreamWaQRecurrentV3SGMA requires symmetry_helper_class_name.")
        self.symmetry_helper_class_name = symmetry_helper_class_name
        self.symmetry_helper = None
        self.policy_normalizer = None
        self.critic_normalizer = None
        self.aug_cenet_hidden_states = None

    def configure_sgma(self, obs_format, symmetry_helper_class_name, policy_normalizer=None, critic_normalizer=None):
        module_name, class_name = symmetry_helper_class_name.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        helper_cls = getattr(module, class_name)
        helper_kwargs = {}
        encoder = getattr(self.cenet, "encoder", self.cenet)
        if hasattr(encoder, "num_history_steps"):
            helper_kwargs["history_length"] = encoder.num_history_steps
        if hasattr(encoder, "obs_layout"):
            helper_kwargs["obs_layout"] = encoder.obs_layout
        self.symmetry_helper = helper_cls(obs_format, **helper_kwargs)
        self.policy_normalizer = policy_normalizer
        self.critic_normalizer = critic_normalizer
        self.aug_cenet_hidden_states = torch.zeros(
            self.cenet.rnn.num_layers,
            self.storage.num_envs,
            self.cenet.rnn.hidden_size,
            device=self.device,
        )

    def _apply_normalizer(self, normalizer, tensor):
        if normalizer is None or tensor is None:
            return tensor
        original_shape = tensor.shape
        normalized = normalizer(tensor.reshape(-1, original_shape[-1]))
        return normalized.reshape(original_shape)

    def _build_offline_augmented_rollout(self):
        if self.symmetry_helper is None:
            raise ValueError("SGMA symmetry helper is not configured.")

        raw_obs = self.storage.get_direct_policy_obs_sequences(self.cenet.raw_encoder_input_dim)
        critic_obs = self.storage.critic_observations
        aug_raw_obs = self.symmetry_helper.mirror_group("policy", raw_obs)
        aug_raw_obs = self._apply_normalizer(self.policy_normalizer, aug_raw_obs)

        aug_critic_obs = None
        if critic_obs is not None:
            aug_critic_obs = self.symmetry_helper.mirror_group("critic", critic_obs)
            aug_critic_obs = self._apply_normalizer(self.critic_normalizer, aug_critic_obs)

        aug_actions = self.symmetry_helper.mirror_actions(self.storage.actions)

        sequence_masks = torch.ones(
            self.storage.num_transitions_per_env,
            self.storage.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        aug_estimates = self.cenet.encode(
            aug_raw_obs,
            hidden_states=self.aug_cenet_hidden_states,
            masks=sequence_masks,
        ).detach()

        hidden_states = self.aug_cenet_hidden_states.clone()
        aug_latents = torch.zeros(
            self.storage.num_transitions_per_env,
            self.storage.num_envs,
            self.cenet.latent_dim,
            device=self.device,
        )
        for step in range(self.storage.num_transitions_per_env):
            if step > 0:
                done_ids = self.storage.dones[step - 1].view(-1).nonzero(as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    hidden_states[:, done_ids] = 0.0

            estimate = aug_estimates[step]
            if self.use_estimate:
                final_estimate = estimate.detach()
            else:
                critic_for_vel = aug_critic_obs[step] if aug_critic_obs is not None else aug_raw_obs[step]
                lin_vel = get_subobs_by_components(
                    critic_for_vel,
                    ["base_lin_vel"],
                    self.actor_critic.critic_obs_segments,
                )
                final_estimate = torch.cat([lin_vel.detach(), estimate[:, 3:].detach()], dim=-1)

            aug_latents[step] = final_estimate
            _, hidden_states = self.cenet.encoder_inference_recurrent(aug_raw_obs[step], hidden_states)
            hidden_states = hidden_states.detach()

        done_ids = self.storage.dones[-1].view(-1).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            hidden_states[:, done_ids] = 0.0
        self.aug_cenet_hidden_states = hidden_states.detach()

        aug_obs = torch.cat([aug_raw_obs, aug_latents], dim=-1)
        self.storage.set_sgma_augmentation(
            aug_observations=aug_obs,
            aug_critic_observations=aug_critic_obs,
            aug_actions=aug_actions,
            aug_old_actions_log_prob=self.storage.actions_log_prob.clone(),
            aug_returns=self.storage.returns.clone(),
            aug_advantages=self.storage.advantages.clone(),
        )

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)
        if minibatch.aug_obs is None:
            return losses, inter_vars, stats

        self.actor_critic.act(minibatch.aug_obs)
        aug_log_prob = self.actor_critic.get_actions_log_prob(minibatch.aug_actions)
        old_aug_log_prob = minibatch.aug_old_actions_log_prob.squeeze(-1)
        mixed_advantages = torch.mean(minibatch.aug_advantages * self.advantage_mixing_weights, dim=-1)
        aug_ratio = torch.exp(aug_log_prob - old_aug_log_prob)
        aug_surrogate = -mixed_advantages * aug_ratio
        aug_surrogate_clipped = -mixed_advantages * torch.clamp(
            aug_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        aug_surrogate_loss = torch.max(aug_surrogate, aug_surrogate_clipped).mean()

        losses["surrogate_loss"] = 0.5 * (losses["surrogate_loss"] + aug_surrogate_loss)
        stats["sgma_ratio_mean"] = aug_ratio.detach().mean()
        stats["sgma_surrogate_loss"] = aug_surrogate_loss.detach()
        stats["sgma_action_mean_abs"] = self.actor_critic.action_mean.detach().abs().mean()
        return losses, inter_vars, stats

    def update(self, current_learning_iteration):
        self._build_offline_augmented_rollout()
        losses, stats = super().update(current_learning_iteration)
        self.storage.clear_sgma_augmentation()
        return losses, stats
