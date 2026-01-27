import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage import RolloutStorage
from instinct_rl.utils.utils import get_subobs_by_components


class RolloutStorageDreamWaQ(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.single_obs = None
            self.rewards_noClip = None

    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorage.MiniBatch._fields,
            "single_obs",
        ],
    )

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, 
                 num_single_obs, num_rewards=1, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, 
                         num_rewards, device)
        
        self.single_obs = torch.zeros(num_transitions_per_env, num_envs, num_single_obs, device=self.device)
        self.rewards_noClip = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)

    def add_transitions(self, transition: Transition):
        self.single_obs[self.step].copy_(transition.single_obs)
        self.rewards_noClip[self.step].copy_(transition.rewards_noClip.view(-1, self.num_rewards))
        super().add_transitions(transition)

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
        
        # Note: Recurrent support for single_obs is not fully implemented/verified here as DreamWaQ mostly uses MLP currently
        if padded_B_slice is None:
            single_obs_batch = self.single_obs[T_select, B_select]
        else:
            # Placeholder for recurrent logic if needed, simplistically taking from stored
             single_obs_batch = self.single_obs[T_select, B_select] # Warning: Might need padding logic if recurrent used

        return RolloutStorageDreamWaQ.MiniBatch(*minibatch, single_obs_batch)


class PPODreamWaQ(PPO):
    def __init__(self,
                 actor_critic,
                 cenet,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 advantage_mixing_weights=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 vae_beta=0.5,
                 use_Adaboot = True,
                 device='cpu',
                 **kwargs):
        
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, 
                         advantage_mixing_weights, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, 
                         use_clipped_value_loss, schedule=schedule, desired_kl=desired_kl, device=device, **kwargs)

        self.cenet = cenet
        self.cenet.to(self.device)
        self.vae_beta = vae_beta
        self.use_Adaboot = use_Adaboot
        self.cenet_loss_list = [torch.tensor(0.0, device=self.device) for _ in range(5)]
        self.Pboot = torch.tensor(1.0, device=self.device)
        self.optimizer_cenet = optim.Adam(self.cenet.parameters(), lr=learning_rate)
        self.use_estimate=False
        
        # Override transition with the custom one
        self.transition = RolloutStorageDreamWaQ.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, num_single_obs=0):
        obs_size = 0
        for k, v in obs_format["policy"].items():
            import numpy as np
            obs_size += np.prod(v)
            
        critic_obs_size = 0
        if "critic" in obs_format:
            for k, v in obs_format["critic"].items():
                import numpy as np
                critic_obs_size += np.prod(v)
        else:
            critic_obs_size = None

        self.storage = RolloutStorageDreamWaQ(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_single_obs,
            num_rewards=num_rewards,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        # Concatenate estimate to obs
        # Note: obs coming in is likely just the raw policy obs from env
        estimate = self.cenet.encode(obs).detach()
        if self.use_estimate:
            final_estimate = estimate
        else:
            lin_vel = get_subobs_by_components(critic_obs, ["base_lin_vel"], 
                                                self.actor_critic.critic_obs_segments)
            z_part = estimate[:, 3:] # 或者 torch.zeros_like(...)
            final_estimate = torch.cat([lin_vel.detach(), z_part], dim=-1)
        obs_augmented = torch.cat((obs, final_estimate), dim= -1)
        
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            
        self.transition.actions = self.actor_critic.act(obs_augmented).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs if critic_obs is not None else obs_augmented).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        self.transition.observations = obs_augmented
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, rewards_noClip, num_single_obs, next_obs, next_critic_obs, next_critic_obs_for_bootstrap=None):
        
        true_next_critic_obs = next_critic_obs.clone()
        
        if "termination_observations" in infos and "critic" in infos["termination_observations"]:
            term_ids = infos["termination_env_ids"]
            if len(term_ids) > 0:
                # 用“临终”观测值覆盖掉“新生”观测值
                true_next_critic_obs[term_ids] = infos["termination_observations"]["critic"]
        self.transition.single_obs = true_next_critic_obs[:,:num_single_obs]
        self.transition.rewards_noClip = rewards_noClip.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs logic (similar to Base PPO)
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

    def compute_Pboot(self):
        update_cenet = True
        if self.use_Adaboot:
            rewards_noClip = self.storage.rewards_noClip.clone()
            # rewards_noClip shape: (steps, envs, rewards)
            # Assuming single reward for now or summing them? DWL does sum dim 1 (steps) after transpose? 
            # DWL: rewards_noClip.squeeze(-1).transpose(0, 1) -> (envs, steps)
            # episodic_rewards = torch.sum(rewards_noClip, dim=1) -> (envs,)
            
            rewards_sum = rewards_noClip.sum(dim=0).squeeze(-1) # Sum over time steps
            
            mean_episodic_rewards = torch.mean(rewards_sum)
            std_episodic_rewards = torch.std(rewards_sum)

            # CV
            cv = std_episodic_rewards / mean_episodic_rewards if mean_episodic_rewards > 0 else torch.tensor(10086.0).to(self.device)

            self.Pboot = 1 - torch.tanh(cv)
            if self.Pboot > torch.rand(1, dtype=torch.float32, device=self.device):
                update_cenet = True
            else:
                update_cenet = False
        return update_cenet

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        
        mean_value_loss = torch.tensor(0.0, device=self.device)
        mean_surrogate_loss = torch.tensor(0.0, device=self.device)
        mean_loss_vt = torch.tensor(0.0, device=self.device)
        mean_loss_ot = torch.tensor(0.0, device=self.device)
        mean_loss_kl = torch.tensor(0.0, device=self.device)
        mean_loss_est = torch.tensor(0.0, device=self.device)
        
        # [修改 1] 移出循环：每轮迭代只计算一次 Pboot 状态
        # 这决定了"下一次" Rollout 时是否使用估测器
        # 显式赋值，处理 True/False 两种情况
        self.use_estimate = self.compute_Pboot() 
        
        # PPO Update Loop
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for minibatch in generator:
            # Base PPO Update (Actor & Critic)
            losses, _, _ = self.compute_losses(minibatch)
            loss = losses["surrogate_loss"] + self.value_loss_coef * losses["value_loss"] + losses.get("entropy", 0.0) * self.entropy_coef
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += losses["value_loss"]
            mean_surrogate_loss += losses["surrogate_loss"]
            
            # --- Estimator Update (始终运行) ---
            
            # 1. 获取 Ground Truth 速度
            lin_vel = get_subobs_by_components(minibatch.critic_obs, ["base_lin_vel"], 
                                                self.actor_critic.critic_obs_segments)
            
            # 2. 剥离旧的 Estimate，获取纯净观测
            # 确保使用正确的维度切片
            raw_obs = minibatch.obs[..., :self.cenet.encoder.model[0].in_features] 
            
            # 3. 前向传播 (Train mode -> 会采样)
            z_sample = self.cenet.encode(raw_obs)
            est_mean = self.cenet.encoder_mean
            est_logvar = self.cenet.encoder_logvar # 注意：这里要是 log_std * 2
            
            # 4. 计算 KL Loss (只针对 Context 部分: 后16维)
            # 确保 est_logvar 是 log(sigma^2)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + est_logvar[:,-16:] - est_mean[:,-16:] ** 2 - torch.exp(est_logvar[:,-16:]), dim=1))            
            
            # 5. 计算功能 Loss
            loss_vt = torch.nn.functional.mse_loss(est_mean[:,:3], lin_vel) # 速度估计用均值
            loss_ot = torch.nn.functional.mse_loss(self.cenet.decode(z_sample), minibatch.single_obs) # 重建用采样值
            
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
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        
        # 这里的统计逻辑不再需要 if cenet_update，因为它总是更新的
        mean_loss_vt /= num_updates
        mean_loss_ot /= num_updates
        mean_loss_kl /= num_updates
        mean_loss_est /= num_updates
        self.cenet_loss_list = [mean_loss_vt, mean_loss_ot, mean_loss_kl, mean_loss_est, self.Pboot]
            
        self.storage.clear()
        
        stats = {
            "estimator_loss": self.cenet_loss_list[3],
            "estimator_mse_vt": self.cenet_loss_list[0],
            "estimator_mse_ot": self.cenet_loss_list[1],
            "estimator_kl": self.cenet_loss_list[2],
            "Pboot": self.Pboot
        }

        mean_losses = {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "estimator_loss": mean_loss_est
        }
        
        return mean_losses, stats
