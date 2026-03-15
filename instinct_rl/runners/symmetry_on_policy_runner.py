from __future__ import annotations

from collections import deque
import time

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.utils.utils import store_code_state


class SymmetryOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that keeps the policy in a unified symmetry frame.

    The second half of the environments is interpreted in a mirrored coordinate frame:
    observations are mirrored before entering the policy, and actions are mirrored back
    before being applied to the environment. This keeps recurrent hidden states aligned
    with a consistent frame over time.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.symmetry_cfg = train_cfg.get("symmetry", None)
        if self.symmetry_cfg is None or not self.symmetry_cfg.get("enabled", False):
            raise ValueError("SymmetryOnPolicyRunner requires symmetry.enabled=True in the runner config.")
        if self.env.num_envs % 2 != 0:
            raise ValueError("SymmetryOnPolicyRunner requires an even number of environments.")
        self._mirror_start = self.env.num_envs // 2
        self.symmetry_helper = self._build_symmetry_helper(self.symmetry_cfg["helper_class_name"])

    def _build_symmetry_helper(self, helper_class_name: str):
        module_name, class_name = helper_class_name.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        helper_cls = getattr(module, class_name)
        return helper_cls(self.env.get_obs_format())

    def _mirror_obs_group_tail(self, group_name: str, obs: torch.Tensor | None) -> torch.Tensor | None:
        if obs is None:
            return None
        mirrored = obs.clone()
        mirrored_tail = self.symmetry_helper.mirror_group(group_name, mirrored[self._mirror_start :])
        mirrored[self._mirror_start :] = mirrored_tail
        return mirrored

    def _mirror_action_tail_to_env(self, actions: torch.Tensor) -> torch.Tensor:
        env_actions = actions.clone()
        env_actions[self._mirror_start :] = self.symmetry_helper.mirror_actions(env_actions[self._mirror_start :])
        return env_actions

    def _mirror_observation_dict(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {group_name: self._mirror_obs_group_tail(group_name, value) for group_name, value in observations.items()}

    def _mirror_termination_observations(self, infos: dict):
        term_obs = infos.get("termination_observations")
        term_env_ids = infos.get("termination_env_ids")
        if not isinstance(term_obs, dict) or term_env_ids is None or len(term_env_ids) == 0:
            return
        mirror_mask = term_env_ids >= self._mirror_start
        if not torch.any(mirror_mask):
            return
        for group_name, value in term_obs.items():
            if not isinstance(value, torch.Tensor):
                continue
            mirrored_value = value.clone()
            mirrored_value[mirror_mask] = self.symmetry_helper.mirror_group(group_name, mirrored_value[mirror_mask])
            term_obs[group_name] = mirrored_value

    def _prepare_policy_observations(self, obs: torch.Tensor, critic_obs: torch.Tensor | None):
        obs_policy = self._mirror_obs_group_tail("policy", obs)
        critic_policy = self._mirror_obs_group_tail("critic", critic_obs) if critic_obs is not None else None
        return obs_policy, critic_policy

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if dist.is_initialized():
            self.alg.distributed_data_parallel()
            print(f"[INFO rank {dist.get_rank()}]: DistributedDataParallel enabled.")
        if self.log_dir is not None and self.writer is None and (not self.is_mp_rank_other_process()):
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        critic_obs = extras["observations"].get("critic", None)
        critic_obs = critic_obs.to(self.device) if critic_obs is not None else None
        obs, critic_obs = self._prepare_policy_observations(obs, critic_obs)
        self.train_mode()

        ep_infos = []
        step_infos = []
        rframebuffer = [deque(maxlen=2000) for _ in range(self.env.num_rewards)]
        rewbuffer = [deque(maxlen=100) for _ in range(self.env.num_rewards)]
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print(
            "[INFO{}]: Initialization done, start learning.".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        print(
            "NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore"
            " it if it does not affect the performance."
        )
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tot_start_time = time.time()
        start = time.time()
        while self.current_learning_iteration < tot_iter:
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for _ in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    if len(rewards.shape) == 1:
                        rewards = rewards.unsqueeze(-1)

                    if self.log_dir is not None:
                        if "step" in infos:
                            step_infos.append(infos["step"])
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
                        for i in range(self.env.num_rewards):
                            rframebuffer[i].extend(rewards[dones < 1][:, i].cpu().numpy().tolist())
                            rewbuffer[i].extend(cur_reward_sum[new_ids][:, i].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)

            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

        self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")

    def rollout_step(self, obs, critic_obs):
        actions_policy = self.alg.act(obs, critic_obs)
        actions_env = self._mirror_action_tail_to_env(actions_policy)

        obs, rewards, dones, infos = self.env.step(actions_env)
        critic_obs = infos["observations"].get("critic", None)
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )

        infos["observations"] = self._mirror_observation_dict(infos["observations"])
        self._mirror_termination_observations(infos)

        obs = infos["observations"]["policy"]
        critic_obs = infos["observations"].get("critic", None)

        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic":
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            else:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])

        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos
