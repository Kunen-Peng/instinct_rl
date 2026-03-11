import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.dreamwaq_v2 import PPODreamWaQRecurrentV2, RolloutStorageDreamWaQRecurrent
from instinct_rl.storage.rollout_storage import RolloutStorage
from instinct_rl.utils import split_and_pad_trajectories
from instinct_rl.utils.buffer import buffer_method
from instinct_rl.utils.utils import get_subobs_by_components


class RolloutStorageDreamWaQRecurrentV3(RolloutStorageDreamWaQRecurrent):
    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorageDreamWaQRecurrent.MiniBatch._fields,
        ],
    )

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        self._padded_obs_trajectories, self._trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.critic_observations is not None:
            self._padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_observations, self.dones)
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
            return super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)

        obs_batch = self._padded_obs_trajectories[T_select, padded_B_slice]
        critic_obs_batch = (
            obs_batch
            if self.critic_observations is None
            else self._padded_critic_obs_trajectories[T_select, padded_B_slice]
        )
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

        return RolloutStorageDreamWaQRecurrentV3.MiniBatch(
            obs_batch,
            critic_obs_batch,
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
        )


class PPODreamWaQRecurrentV3(PPODreamWaQRecurrentV2):
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, num_single_obs=0):
        if self.cenet.rnn is None:
            raise ValueError("PPODreamWaQRecurrentV3 requires a recurrent CENet (rnn is None).")

        cenet_hidden_state_shape = (self.cenet.rnn.num_layers, self.cenet.rnn.hidden_size)

        obs_size = 0
        for _, v in obs_format["policy"].items():
            import numpy as np

            obs_size += np.prod(v)

        critic_obs_size = 0
        if "critic" in obs_format:
            for _, v in obs_format["critic"].items():
                import numpy as np

                critic_obs_size += np.prod(v)
        else:
            critic_obs_size = None

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

    def _compute_recurrent_estimator_losses(self, minibatch):
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
        loss_ot = F.mse_loss(est_obs[valid], minibatch.single_obs[valid])
        return loss_vt, loss_ot, kl_loss

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration

        mean_value_loss = torch.tensor(0.0, device=self.device)
        mean_surrogate_loss = torch.tensor(0.0, device=self.device)
        mean_loss_vt = torch.tensor(0.0, device=self.device)
        mean_loss_ot = torch.tensor(0.0, device=self.device)
        mean_loss_kl = torch.tensor(0.0, device=self.device)
        mean_loss_est = torch.tensor(0.0, device=self.device)

        self.use_estimate = self.compute_Pboot()

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

            mean_value_loss += losses["value_loss"]
            mean_surrogate_loss += losses["surrogate_loss"]

        estimator_generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in estimator_generator:
            for _ in range(self.num_estimator_epochs):
                loss_vt, loss_ot, kl_loss = self._compute_recurrent_estimator_losses(minibatch)
                cenet_loss = loss_vt + loss_ot + self.vae_beta * kl_loss

                self.optimizer_cenet.zero_grad()
                cenet_loss.backward()
                nn.utils.clip_grad_norm_(self.cenet.parameters(), self.max_grad_norm)
                self.optimizer_cenet.step()

                mean_loss_vt += loss_vt
                mean_loss_ot += loss_ot
                mean_loss_kl += kl_loss
                mean_loss_est += cenet_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        est_updates = num_updates * self.num_estimator_epochs

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_loss_vt /= est_updates
        mean_loss_ot /= est_updates
        mean_loss_kl /= est_updates
        mean_loss_est /= est_updates

        k_beta = math.exp(self.beta_delta * (self.target_mse_ot - mean_loss_ot.item()))
        self.vae_beta = max(self.min_vae_beta, min(self.max_vae_beta, self.vae_beta * k_beta))

        self.cenet_loss_list = [mean_loss_vt, mean_loss_ot, mean_loss_kl, mean_loss_est, self.Pboot]
        self.storage.clear()

        stats = {
            "estimator_loss": self.cenet_loss_list[3],
            "estimator_mse_vt": self.cenet_loss_list[0],
            "estimator_mse_ot": self.cenet_loss_list[1],
            "estimator_kl": self.cenet_loss_list[2],
            "Pboot": self.Pboot,
        }
        mean_losses = {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "estimator_loss": mean_loss_est,
        }
        return mean_losses, stats
