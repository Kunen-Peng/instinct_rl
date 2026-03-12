import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.dreamwaq_common import (
    PPODreamWaQCommon,
    PPODreamWaQRecurrentCommon,
    RolloutStorageDreamWaQ,
    RolloutStorageDreamWaQRecurrent,
)
from instinct_rl.utils.utils import get_subobs_by_components


class PPODreamWaQV2(PPODreamWaQCommon):
    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        accumulators = self._init_update_stats()
        self.use_estimate = self.compute_use_estimate()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:
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

            lin_vel = get_subobs_by_components(minibatch.critic_obs, ["base_lin_vel"], self.actor_critic.critic_obs_segments)
            raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
            valid = ~minibatch.dones.squeeze(-1).bool()

            for _ in range(self.num_estimator_epochs):
                z_sample = self.cenet.encode(raw_obs)
                est_mean = self.cenet.encoder_mean
                est_logvar = self.cenet.encoder_logvar
                kl_loss = torch.mean(
                    -0.5
                    * torch.sum(
                        1 + est_logvar[valid] - est_mean[valid, -16:] ** 2 - torch.exp(est_logvar[valid]),
                        dim=1,
                    )
                )
                loss_vt = F.mse_loss(est_mean[valid, :3], lin_vel[valid])
                est_obs = self.cenet.decode(z_sample)
                loss_ot = F.mse_loss(est_obs[valid], minibatch.single_obs[valid])
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


class PPODreamWaQRecurrentV2(PPODreamWaQRecurrentCommon):
    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        accumulators = self._init_update_stats()
        self.use_estimate = self.compute_use_estimate()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:
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

            lin_vel = get_subobs_by_components(minibatch.critic_obs, ["base_lin_vel"], self.actor_critic.critic_obs_segments)
            raw_obs = minibatch.obs[..., : self.cenet.raw_encoder_input_dim]
            valid = ~minibatch.dones.squeeze(-1).bool()
            h_0 = minibatch.cenet_hidden_states.permute(1, 0, 2).contiguous()

            for _ in range(self.num_estimator_epochs):
                z_sample = self.cenet.encode(raw_obs, hidden_states=h_0)
                est_mean = self.cenet.encoder_mean
                est_logvar = self.cenet.encoder_logvar
                kl_loss = torch.mean(
                    -0.5
                    * torch.sum(
                        1 + est_logvar[valid] - est_mean[valid, -16:] ** 2 - torch.exp(est_logvar[valid]),
                        dim=1,
                    )
                )
                loss_vt = F.mse_loss(est_mean[valid, :3], lin_vel[valid])
                est_obs = self.cenet.decode(z_sample)
                loss_ot = F.mse_loss(est_obs[valid], minibatch.single_obs[valid])
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
