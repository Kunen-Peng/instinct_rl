from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.dreamwaq_np3o_common import (
    DreamWaQNP3ORecurrentCommon,
    RolloutStorageDreamWaQWithCostRecurrent,
)
from instinct_rl.storage.rollout_storage import RolloutStorage
from instinct_rl.utils import split_and_pad_trajectories
from instinct_rl.utils.buffer import buffer_method
from instinct_rl.utils.utils import get_subobs_by_components


class RolloutStorageDreamWaQWithCostRecurrentV3(RolloutStorageDreamWaQWithCostRecurrent):
    MiniBatch = namedtuple("MiniBatch", [*RolloutStorageDreamWaQWithCostRecurrent.MiniBatch._fields])

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        self._padded_obs_trajectories, self._trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.critic_observations is not None:
            self._padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_observations, self.dones)
        self._padded_single_obs_trajectories, _ = split_and_pad_trajectories(self.single_obs, self.dones)
        self._padded_cost_values, _ = split_and_pad_trajectories(self.cost_values, self.dones)
        self._padded_cost_advantages, _ = split_and_pad_trajectories(self.cost_advantages, self.dones)
        self._padded_cost_returns, _ = split_and_pad_trajectories(self.cost_returns, self.dones)
        self._padded_cost_violation, _ = split_and_pad_trajectories(self.cost_violation, self.dones)

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

                minibatch = self.get_minibatch_from_selection(
                    slice(None),
                    slice(start, stop),
                    padded_B_slice=slice(first_traj, last_traj),
                    prev_done_mask=last_was_done,
                )
                yield (
                    minibatch,
                    self._padded_cost_values[:, first_traj:last_traj],
                    self._padded_cost_advantages[:, first_traj:last_traj],
                    self._padded_cost_returns[:, first_traj:last_traj],
                    self._padded_cost_violation[:, first_traj:last_traj],
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
            return super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)

        obs_batch = self._padded_obs_trajectories[T_select, padded_B_slice]
        critic_obs_batch = (
            obs_batch if self.critic_observations is None else self._padded_critic_obs_trajectories[T_select, padded_B_slice]
        )
        obs_mask_batch = self._trajectory_masks[T_select, padded_B_slice]
        single_obs_batch = self._padded_single_obs_trajectories[T_select, padded_B_slice]
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

        return RolloutStorageDreamWaQWithCostRecurrentV3.MiniBatch(
            obs_batch,
            critic_obs_batch,
            self.actions[T_select, B_select],
            self.values[T_select, B_select],
            self.advantages[T_select, B_select],
            self.returns[T_select, B_select],
            self.actions_log_prob[T_select, B_select],
            self.mu[T_select, B_select],
            self.sigma[T_select, B_select],
            hid_batch,
            obs_mask_batch,
            single_obs_batch,
            cenet_hidden_states_batch,
        )


class DreamWaQNP3OV3(DreamWaQNP3ORecurrentCommon):
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
        obs_size = sum(__import__("numpy").prod(v) for v in obs_format["policy"].values())
        critic_obs_size = (
            sum(__import__("numpy").prod(v) for v in obs_format["critic"].values()) if "critic" in obs_format else None
        )
        if cost_shape is None:
            raise ValueError("cost_shape must be provided for DreamWaQNP3OV3")
        self.k_value = self._to_cost_tensor(self.k_value, cost_shape)

        if self.cenet.rnn is not None:
            cenet_hidden_state_shape = (self.cenet.rnn.num_layers, self.cenet.rnn.hidden_size)
            self.storage = RolloutStorageDreamWaQWithCostRecurrentV3(
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
        else:
            # MLP mode — delegate to base class (non-recurrent storage).
            super().init_storage(
                num_envs, num_transitions_per_env, obs_format, num_actions,
                num_rewards=num_rewards, cost_shape=cost_shape,
                cost_d_values=cost_d_values, num_single_obs=num_single_obs,
            )

    def _compute_recurrent_estimator_losses(self, minibatch):
        """Compute estimator losses over padded trajectory batches (RNN only)."""
        lin_vel = get_subobs_by_components(minibatch.critic_obs, ["base_lin_vel"], self._critic_obs_segments)
        raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
        valid = minibatch.masks.bool()
        z_sample = self.cenet.encode(raw_obs, hidden_states=minibatch.cenet_hidden_states, masks=minibatch.masks)
        est_mean = self.cenet.encoder_mean
        est_logvar = self.cenet.encoder_logvar
        est_obs = self.cenet.decode(z_sample)
        z_mean = est_mean[..., self.cenet.dim_v :]
        kl_terms = -0.5 * torch.sum(1 + est_logvar - z_mean.square() - torch.exp(est_logvar), dim=-1)
        return (
            F.mse_loss(est_mean[..., : self.cenet.dim_v][valid], lin_vel[valid]),
            F.mse_loss(est_obs[valid], minibatch.single_obs[valid]),
            kl_terms[valid].mean(),
        )

    def _compute_mlp_estimator_losses(self, minibatch):
        """Compute estimator losses over flat mini-batches (MLP only)."""
        lin_vel = get_subobs_by_components(minibatch.critic_obs, ["base_lin_vel"], self._critic_obs_segments)
        raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
        valid = ~minibatch.dones.squeeze(-1).bool() if hasattr(minibatch, 'dones') else torch.ones(raw_obs.shape[:-1], dtype=torch.bool, device=raw_obs.device)
        z_sample = self.cenet.encode(raw_obs)
        est_mean = self.cenet.encoder_mean
        est_logvar = self.cenet.encoder_logvar
        est_obs = self.cenet.decode(z_sample)
        z_mean = est_mean[..., self.cenet.dim_v :]
        kl_terms = -0.5 * torch.sum(1 + est_logvar - z_mean.square() - torch.exp(est_logvar), dim=-1)
        return (
            F.mse_loss(est_mean[..., : self.cenet.dim_v][valid], lin_vel[valid]),
            F.mse_loss(est_obs[valid], minibatch.single_obs[valid]),
            kl_terms[valid].mean(),
        )

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        estimator_stats = self._init_estimator_stats()
        self.use_estimate = self.compute_use_estimate()

        ppo_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch_data in ppo_generator:
            minibatch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch = minibatch_data
            losses, _, stats = self.compute_losses(minibatch)
            cost_value_loss = self._compute_cost_value_loss(minibatch, target_cost_values_batch, cost_returns_batch)
            viol_loss = self._compute_penalty_loss(minibatch, cost_advantages_batch, cost_violation_batch)
            total_ac_loss = 0.0
            for k, v in losses.items():
                total_ac_loss = total_ac_loss + getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] += v.detach()
            total_ac_loss = total_ac_loss + self.cost_value_loss_coef * cost_value_loss + self.cost_viol_loss_coef * viol_loss
            mean_losses["cost_value_loss"] += cost_value_loss.detach()
            mean_losses["viol_loss"] += viol_loss.detach()
            mean_losses["total_ac_loss"] += total_ac_loss.detach()

            self.optimizer.zero_grad()
            total_ac_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._accumulate_stats(average_stats, stats)

        if self.cenet.rnn is not None:
            # RNN mode: use recurrent mini-batch generator with padded trajectories.
            estimator_generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for minibatch_data in estimator_generator:
                minibatch = minibatch_data[0]
                for _ in range(self.num_estimator_epochs):
                    loss_vt, loss_ot, kl_loss = self._compute_recurrent_estimator_losses(minibatch)
                    cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss
                    self.optimizer_cenet.zero_grad()
                    cenet_loss.backward()
                    nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
                    self.optimizer_cenet.step()
                    estimator_stats["loss_vt"] += loss_vt.detach()
                    estimator_stats["loss_ot"] += loss_ot.detach()
                    estimator_stats["loss_kl"] += kl_loss.detach()
                    estimator_stats["loss_est"] += cenet_loss.detach()
        else:
            # MLP mode: use standard mini-batch generator.
            est_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            for minibatch_data in est_generator:
                minibatch = minibatch_data[0]
                for _ in range(self.num_estimator_epochs):
                    loss_vt, loss_ot, kl_loss = self._compute_mlp_estimator_losses(minibatch)
                    cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss
                    self.optimizer_cenet.zero_grad()
                    cenet_loss.backward()
                    nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
                    self.optimizer_cenet.step()
                    estimator_stats["loss_vt"] += loss_vt.detach()
                    estimator_stats["loss_ot"] += loss_ot.detach()
                    estimator_stats["loss_kl"] += kl_loss.detach()
                    estimator_stats["loss_est"] += cenet_loss.detach()

        return self._finalize_estimator_stats(estimator_stats, mean_losses, average_stats)

