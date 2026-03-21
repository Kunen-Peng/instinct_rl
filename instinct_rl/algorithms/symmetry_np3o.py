from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from .np3o import NP3O


class SymmetryNP3O(NP3O):
    """NP3O with update-stage symmetry augmentation for feedforward policies."""

    def __init__(
        self,
        actor_critic,
        symmetry_helper_class_name: str | None = None,
        symmetry_use_critic_augmentation: bool = False,
        symmetry_use_mirror_consistency_loss: bool = False,
        symmetry_loss_coef: float = 0.0,
        symmetry_warn_large_init_scale: bool = True,
        symmetry_large_init_std_threshold: float = 1.0,
        **kwargs,
    ):
        super().__init__(actor_critic, **kwargs)
        if self.actor_critic.is_recurrent:
            raise ValueError(
                "SymmetryNP3O only supports feedforward actor-critic models. "
                "Use a symmetry runner for recurrent DreamWaQ-NP3O branches."
            )
        if symmetry_helper_class_name is None:
            raise ValueError("SymmetryNP3O requires symmetry_helper_class_name to build the symmetry helper.")

        self.symmetry_use_critic_augmentation = symmetry_use_critic_augmentation
        self.symmetry_use_mirror_consistency_loss = symmetry_use_mirror_consistency_loss
        self.symmetry_loss_coef = symmetry_loss_coef
        self.symmetry_warn_large_init_scale = symmetry_warn_large_init_scale
        self.symmetry_large_init_std_threshold = symmetry_large_init_std_threshold
        self.symmetry_helper = self._build_symmetry_helper(symmetry_helper_class_name)
        self.policy_normalizer = None
        self.critic_normalizer = None
        self._warn_if_large_initialization_scale()

    def _build_symmetry_helper(self, helper_class_name: str):
        module_name, class_name = helper_class_name.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        helper_cls = getattr(module, class_name)
        obs_format = {
            "policy": self.actor_critic.obs_segments,
            "critic": self.actor_critic.critic_obs_segments,
        }
        return helper_cls(obs_format)

    def configure_normalizers(self, policy_normalizer=None, critic_normalizer=None):
        self.policy_normalizer = policy_normalizer
        self.critic_normalizer = critic_normalizer

    def _apply_normalizer(self, normalizer, observations: torch.Tensor | None) -> torch.Tensor | None:
        if observations is None or normalizer is None:
            return observations
        return normalizer(observations)

    def _policy_mean(self, observations: torch.Tensor) -> torch.Tensor:
        self.actor_critic.update_distribution(observations)
        return self.actor_critic.action_mean

    def _warn_if_large_initialization_scale(self):
        if not self.symmetry_warn_large_init_scale:
            return
        if not hasattr(self.actor_critic, "std"):
            return

        init_std_max = self.actor_critic.std.detach().abs().max().item()
        if init_std_max > self.symmetry_large_init_std_threshold:
            warnings.warn(
                "SymmetryNP3O uses a Mittal-style approximation that omits the state-visitation ratio term. "
                f"Observed initial policy std {init_std_max:.4f} exceeds the configured threshold "
                f"{self.symmetry_large_init_std_threshold:.4f}; consider a smaller initialization scale or "
                "tighter policy updates.",
                stacklevel=2,
            )

    def _compute_value_loss(self, value_batch, value_targets, value_returns):
        if self.use_clipped_value_loss:
            value_clipped = value_targets + (value_batch - value_targets).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - value_returns).pow(2)
            value_losses_clipped = (value_clipped - value_returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = (value_returns - value_batch).pow(2)
        return value_loss.reshape(-1, value_loss.shape[-1]).mean(dim=0)

    def _get_augmented_batch(self, minibatch):
        raw_obs = minibatch.raw_obs if minibatch.raw_obs is not None else minibatch.obs
        raw_critic_obs = minibatch.raw_critic_obs if minibatch.raw_critic_obs is not None else minibatch.critic_obs

        obs_aug = self.symmetry_helper.mirror_group("policy", raw_obs)
        obs_aug = self._apply_normalizer(self.policy_normalizer, obs_aug)

        critic_source = raw_critic_obs if raw_critic_obs is not None else raw_obs
        critic_obs_aug = self.symmetry_helper.mirror_group("critic", critic_source)
        critic_obs_aug = self._apply_normalizer(self.critic_normalizer, critic_obs_aug)

        actions_aug = self.symmetry_helper.mirror_actions(minibatch.actions)
        old_log_prob_batch = minibatch.old_actions_log_prob.squeeze(-1)
        return obs_aug, critic_obs_aug, actions_aug, old_log_prob_batch

    def compute_losses(self, minibatch):
        if self.actor_critic.is_recurrent:
            raise ValueError(
                "SymmetryNP3O only supports feedforward actor-critic models. "
                "Use a symmetry runner for recurrent DreamWaQ-NP3O branches."
            )

        self.actor_critic.update_distribution(minibatch.obs)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        critic_obs = minibatch.critic_obs if minibatch.critic_obs is not None else minibatch.obs
        value_batch = self.actor_critic.evaluate(critic_obs)
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

        old_log_prob_batch = minibatch.old_actions_log_prob.squeeze(-1)
        mixed_advantages = torch.mean(minibatch.advantages * self.advantage_mixing_weights, dim=-1)

        ratio = torch.exp(actions_log_prob_batch - old_log_prob_batch)
        surrogate = -mixed_advantages * ratio
        surrogate_clipped = -mixed_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_loss = self._compute_value_loss(value_batch, minibatch.values, minibatch.returns)

        obs_aug, critic_obs_aug, actions_aug, old_log_prob_batch = self._get_augmented_batch(minibatch)

        self.actor_critic.update_distribution(obs_aug)
        new_log_prob_aug = self.actor_critic.get_actions_log_prob(actions_aug)
        mu_aug = self.actor_critic.action_mean
        sigma_aug = self.actor_critic.action_std
        ratio_aug = torch.exp(new_log_prob_aug - old_log_prob_batch)
        surrogate_aug = -mixed_advantages * ratio_aug
        surrogate_aug_clipped = -mixed_advantages * torch.clamp(
            ratio_aug, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_aug_loss = torch.max(surrogate_aug, surrogate_aug_clipped).mean()
        surrogate_loss = 0.5 * (surrogate_loss + surrogate_aug_loss)

        augmentation_value_loss = None
        if self.symmetry_use_critic_augmentation:
            value_aug = self.actor_critic.evaluate(critic_obs_aug if critic_obs_aug is not None else obs_aug)
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

        stats_ = {
            "symmetry_ratio_mean": ratio_aug.detach().mean(),
            "symmetry_surrogate_loss": surrogate_aug_loss.detach(),
            "symmetry_action_mean_abs": mu_aug.detach().abs().mean(),
        }
        if value_loss.numel() > 1:
            for i in range(minibatch.advantages.shape[-1]):
                stats_[f"advantage_{i}"] = minibatch.advantages[..., i].detach().mean()
            for i in range(value_loss.numel()):
                stats_[f"value_loss_{i}"] = value_loss.detach().cpu()[i]
        with torch.inference_mode():
            symmetry_kl = torch.sum(
                torch.log(sigma_aug / minibatch.old_sigma + 1.0e-5)
                + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_aug))
                / (2.0 * torch.square(sigma_aug))
                - 0.5,
                axis=-1,
            )
        stats_["symmetry_kl_mean"] = symmetry_kl.detach().mean()
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

    def _compute_cost_value_loss(self, minibatch, target_cost_values_batch, cost_returns_batch):
        cost_value_batch = self._evaluate_cost(minibatch.critic_obs if minibatch.critic_obs is not None else minibatch.obs)

        if not self.use_clipped_value_loss:
            value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()
        else:
            cost_value_clipped = target_cost_values_batch + (
                cost_value_batch - target_cost_values_batch
            ).clamp(-self.clip_param, self.clip_param)
            cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
            cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
            value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()

        if not self.symmetry_use_critic_augmentation:
            return value_loss

        obs_aug, critic_obs_aug, _, _ = self._get_augmented_batch(minibatch)
        cost_value_aug = self._evaluate_cost(critic_obs_aug if critic_obs_aug is not None else obs_aug)
        aug_value_loss = (cost_value_aug - cost_returns_batch).pow(2).mean()
        return 0.5 * (value_loss + aug_value_loss)

    def _compute_penalty_loss(self, minibatch, cost_advantages_batch, cost_violation_batch):
        self.actor_critic.update_distribution(minibatch.obs)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        base_viol = self.compute_viol(
            actions_log_prob_batch,
            minibatch.old_actions_log_prob,
            cost_advantages_batch,
            cost_violation_batch,
        )

        obs_aug, _, actions_aug, _ = self._get_augmented_batch(minibatch)
        self.actor_critic.update_distribution(obs_aug)
        new_log_prob_aug = self.actor_critic.get_actions_log_prob(actions_aug)
        aug_viol = self.compute_viol(
            new_log_prob_aug,
            minibatch.old_actions_log_prob,
            cost_advantages_batch,
            cost_violation_batch,
        )
        return 0.5 * (base_viol + aug_viol)
