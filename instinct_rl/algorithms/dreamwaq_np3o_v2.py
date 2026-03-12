from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.dreamwaq_np3o_common import DreamWaQNP3OCommon, RolloutStorageDreamWaQWithCost
from instinct_rl.utils.utils import get_subobs_by_components


class DreamWaQNP3OV2(DreamWaQNP3OCommon):
    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        estimator_stats = self._init_estimator_stats()
        self.use_estimate = self.compute_use_estimate()

        generator = self._get_batch_generator()
        for minibatch_data in generator:
            minibatch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch = minibatch_data
            losses, _, stats = self.compute_losses(minibatch)
            cost_value_loss = self._compute_cost_value_loss(minibatch, target_cost_values_batch, cost_returns_batch)
            viol_loss = self._compute_penalty_loss(minibatch, cost_advantages_batch, cost_violation_batch)

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

            self.optimizer.zero_grad()
            total_ac_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._accumulate_stats(average_stats, stats)

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

                estimator_stats["loss_vt"] += loss_vt.detach()
                estimator_stats["loss_ot"] += loss_ot.detach()
                estimator_stats["loss_kl"] += kl_loss.detach()
                estimator_stats["loss_est"] += cenet_loss.detach()

        return self._finalize_estimator_stats(estimator_stats, mean_losses, average_stats)
