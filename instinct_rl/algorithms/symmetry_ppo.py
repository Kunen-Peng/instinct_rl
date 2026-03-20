from __future__ import annotations

import torch
import torch.nn.functional as F

from .ppo import PPO


class SymmetryPPO(PPO):
    """PPO with Mittal-style symmetry augmentation applied during the update stage."""

    def __init__(
        self,
        actor_critic,
        symmetry_helper_class_name: str | None = None,
        symmetry_use_critic_augmentation: bool = False,
        symmetry_use_mirror_consistency_loss: bool = False,
        symmetry_loss_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(actor_critic, **kwargs)
        if self.actor_critic.is_recurrent:
            raise ValueError(
                "SymmetryPPO only supports feedforward actor-critic models. "
                "Use SymmetryOnPolicyRunner for recurrent / DreamWaQ v3 branches."
            )
        if symmetry_helper_class_name is None:
            raise ValueError("SymmetryPPO requires symmetry_helper_class_name to build the symmetry helper.")

        self.symmetry_use_critic_augmentation = symmetry_use_critic_augmentation
        self.symmetry_use_mirror_consistency_loss = symmetry_use_mirror_consistency_loss
        self.symmetry_loss_coef = symmetry_loss_coef
        self.symmetry_helper = self._build_symmetry_helper(symmetry_helper_class_name)

    def _build_symmetry_helper(self, helper_class_name: str):
        module_name, class_name = helper_class_name.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        helper_cls = getattr(module, class_name)
        obs_format = {
            "policy": self.actor_critic.obs_segments,
            "critic": self.actor_critic.critic_obs_segments,
        }
        return helper_cls(obs_format)

    def _compute_value_loss(self, value_batch, value_targets, value_returns):
        if self.use_clipped_value_loss:
            value_clipped = value_targets + (value_batch - value_targets).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - value_returns).pow(2)
            value_losses_clipped = (value_clipped - value_returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = (value_returns - value_batch).pow(2)
        return value_loss.reshape(-1, value_loss.shape[-1]).mean(dim=0)

    def _policy_mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.actor(observations)

    def compute_losses(self, minibatch):
        if self.actor_critic.is_recurrent:
            raise ValueError(
                "SymmetryPPO only supports feedforward actor-critic models. "
                "Use SymmetryOnPolicyRunner for recurrent / DreamWaQ v3 branches."
            )

        self.actor_critic.update_distribution(minibatch.obs)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch = self.actor_critic.evaluate(minibatch.critic_obs)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / minibatch.old_sigma + 1.0e-5)
                    + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= torch.distributed.get_world_size()

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                if torch.distributed.is_initialized():
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        old_log_prob_batch = torch.squeeze(minibatch.old_actions_log_prob)
        mixed_advantages = torch.mean(minibatch.advantages * self.advantage_mixing_weights, dim=-1)

        ratio = torch.exp(actions_log_prob_batch - old_log_prob_batch)
        surrogate = -mixed_advantages * ratio
        surrogate_clipped = -mixed_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_loss = self._compute_value_loss(value_batch, minibatch.values, minibatch.returns)

        obs_aug = self.symmetry_helper.mirror_group("policy", minibatch.obs)
        critic_obs_aug = self.symmetry_helper.mirror_group("critic", minibatch.critic_obs)
        actions_aug = self.symmetry_helper.mirror_actions(minibatch.actions)

        self.actor_critic.update_distribution(obs_aug)
        # This avoids naive off-policy probability usage criticized by Mittal 2024.
        new_log_prob_aug = self.actor_critic.get_actions_log_prob(actions_aug)
        ratio_aug = torch.exp(new_log_prob_aug - old_log_prob_batch)
        surrogate_aug = -mixed_advantages * ratio_aug
        surrogate_aug_clipped = -mixed_advantages * torch.clamp(
            ratio_aug, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_aug_loss = torch.max(surrogate_aug, surrogate_aug_clipped).mean()
        surrogate_loss = 0.5 * (surrogate_loss + surrogate_aug_loss)

        augmentation_value_loss = None
        if self.symmetry_use_critic_augmentation:
            value_aug = self.actor_critic.evaluate(critic_obs_aug)
            # The augmented critic target reuses the original returns directly.
            augmentation_value_loss = (value_aug - minibatch.returns).pow(2).reshape(-1, value_aug.shape[-1]).mean(dim=0)
            value_loss = 0.5 * (value_loss + augmentation_value_loss)

        losses = {
            "surrogate_loss": surrogate_loss,
            "value_loss": value_loss.mean(),
            "entropy": -entropy_batch.mean(),
        }

        symmetry_loss = None
        if self.symmetry_use_mirror_consistency_loss:
            mu_orig = self._policy_mean(minibatch.obs)
            mu_aug_pred = self._policy_mean(obs_aug)
            mu_aug_target = self.symmetry_helper.mirror_actions(mu_orig).detach()
            symmetry_loss = F.mse_loss(mu_aug_pred, mu_aug_target)
            losses["symmetry_loss"] = symmetry_loss

        stats_ = {}
        if value_loss.numel() > 1:
            for i in range(minibatch.advantages.shape[-1]):
                stats_[f"advantage_{i}"] = minibatch.advantages[..., i].detach().mean()
            for i in range(value_loss.numel()):
                stats_[f"value_loss_{i}"] = value_loss.detach().cpu()[i]

        stats_.update(
            {
                "symmetry_ratio_mean": ratio_aug.detach().mean(),
                "symmetry_surrogate_loss": surrogate_aug_loss.detach(),
            }
        )
        if augmentation_value_loss is not None:
            stats_["symmetry_value_loss"] = augmentation_value_loss.mean().detach()
        if symmetry_loss is not None:
            stats_["symmetry_loss"] = symmetry_loss.detach()

        inter_vars = {
            "ratio": ratio,
            "surrogate": surrogate,
            "surrogate_clipped": surrogate_clipped,
            "ratio_aug": ratio_aug,
            "surrogate_aug": surrogate_aug,
            "surrogate_aug_clipped": surrogate_aug_clipped,
        }
        if self.desired_kl is not None and self.schedule == "adaptive":
            inter_vars["kl"] = kl

        return losses, inter_vars, stats_
