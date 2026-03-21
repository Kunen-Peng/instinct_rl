import importlib
import os
import time
from collections import deque

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import instinct_rl
import instinct_rl.algorithms as algorithms
from instinct_rl.modules import CENet
from instinct_rl.runners.on_constraint_policy_runner import OnConstraintPolicyRunner
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class DreamWaQNP3ORunner(OnConstraintPolicyRunner):
    """Runner for DreamWaQ-NP3O variants."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.symmetry_cfg = train_cfg.get("symmetry", None)

        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs", self.env.num_obs))
        if num_single_obs is None:
            raise ValueError("DreamWaQNP3ORunner requires num_single_obs to be defined in Env or Runner Config")

        cenet_cfg = {}
        if "cenet" in self.alg_cfg:
            cenet_cfg = self.alg_cfg.pop("cenet")
        elif "cenet" in self.cfg:
            cenet_cfg = self.cfg["cenet"]
        self.cenet = CENet(num_encoder_obs, num_single_obs, **cenet_cfg).to(self.device)

        obs_format = env.get_obs_format()
        est_dim = self.cenet.latent_dim
        if "policy" not in obs_format:
            obs_format["policy"] = {}
        obs_format["policy"]["estimator"] = (est_dim,)

        cost_shape = self._get_cost_shape()
        cost_d_values = self._get_cost_d_values(cost_shape)
        num_costs = cost_shape[0] if cost_shape else 0

        import instinct_rl.modules as modules

        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
            num_costs=num_costs,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        if ":" in alg_class_name:
            module_name, class_name = alg_class_name.rsplit(":", 1)
            alg_class = getattr(importlib.import_module(module_name), class_name)
        else:
            alg_class = getattr(algorithms, alg_class_name)

        self._configure_initial_k_value()
        self.alg = alg_class(actor_critic=actor_critic, cenet=self.cenet, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=(
                    get_subobs_size(obs_format[obs_group_name]) - est_dim
                    if obs_group_name == "policy"
                    else get_subobs_size(obs_format[obs_group_name])
                ),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            num_single_obs=num_single_obs,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [instinct_rl.__file__]
        self._cost_buffer = deque(maxlen=100)
        self.cenet_hidden_states = None

        _, _ = self.env.reset()

    def rollout_step(self, obs, critic_obs):
        if getattr(self.cenet, "rnn", None) is not None:
            act_output = self.alg.act(obs, critic_obs, self.cenet_hidden_states)
        else:
            act_output = self.alg.act(obs, critic_obs)
        if isinstance(act_output, tuple):
            actions, next_cenet_hidden_states = act_output
            self.cenet_hidden_states = next_cenet_hidden_states.detach()
        else:
            actions = act_output
        next_obs, rewards, dones, infos = self.env.step(actions)
        costs = self._extract_costs(infos)

        if self.cenet_hidden_states is not None:
            env_ids_done = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids_done) > 0:
                self.cenet_hidden_states[:, env_ids_done] = 0.0

        next_critic_obs = infos["observations"].get("critic", None)
        next_obs, next_critic_obs, rewards, dones = (
            next_obs.to(self.device),
            next_critic_obs.to(self.device) if next_critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )

        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                next_obs = normalizer(next_obs)
                infos["observations"]["policy"] = next_obs
            elif obs_group_name == "critic":
                if next_critic_obs is not None:
                    next_critic_obs = normalizer(next_critic_obs)
                    infos["observations"]["critic"] = next_critic_obs
            elif obs_group_name in infos["observations"]:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])

        if "termination_observations" in infos:
            term_obs = infos["termination_observations"]
            if isinstance(term_obs, dict):
                for k, v in term_obs.items():
                    if k in self.normalizers:
                        term_obs[k] = self.normalizers[k](v)
            else:
                if "critic" in self.normalizers:
                    infos["termination_observations"] = self.normalizers["critic"](term_obs)

        rewards_noClip = infos.get("rewards_noClip", rewards)
        if isinstance(rewards_noClip, torch.Tensor):
            rewards_noClip = rewards_noClip.to(self.device)

        num_single_obs = self.cenet.decoder.model[-1].out_features
        self.alg.process_env_step(
            rewards=rewards,
            costs=costs,
            dones=dones,
            infos=infos,
            rewards_noClip=rewards_noClip,
            num_single_obs=num_single_obs,
            next_obs=next_obs,
            next_critic_obs=next_critic_obs,
        )

        if costs is not None and self.log_dir is not None:
            self._cost_buffer.extend(costs.sum(dim=-1).cpu().numpy().tolist())

        return next_obs, next_critic_obs, rewards, dones, infos

    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        run_state_dict["cenet_state_dict"] = self.cenet.state_dict()
        run_state_dict["optimizer_cenet_state_dict"] = self.alg.optimizer_cenet.state_dict()
        run_state_dict.update(
            {f"{group_name}_normalizer_state_dict": normalizer.state_dict() for group_name, normalizer in self.normalizers.items()}
        )
        run_state_dict.update({"iter": self.current_learning_iteration, "infos": infos})
        torch.save(run_state_dict, path)

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.load_state_dict(loaded_dict)

        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
        if "optimizer_cenet_state_dict" in loaded_dict:
            self.alg.optimizer_cenet.load_state_dict(loaded_dict["optimizer_cenet_state_dict"])

        for group_name, normalizer in self.normalizers.items():
            key = f"{group_name}_normalizer_state_dict"
            if key not in loaded_dict:
                print(
                    f"\033[1;36m Warning, normalizer for {group_name} is not found, the state dict is not loaded"
                    " \033[0m"
                )
            else:
                normalizer.load_state_dict(loaded_dict[key])

        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.cenet.to(device)

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)

        def policy(obs):
            obs_norm = self.normalizers["policy"](obs) if "policy" in self.normalizers else obs
            latent = self.cenet.encoder_inference(obs_norm)
            obs_aug = torch.cat((obs_norm, latent), dim=-1)
            return self.alg.actor_critic.act_inference(obs_aug)

        return policy

    def export_as_onnx(self, obs, export_model_dir, filename="policy.onnx"):
        self.eval_mode()
        policy_normalizer = self.normalizers.get("policy")
        cenet = self.cenet
        actor_critic = self.alg.actor_critic

        if policy_normalizer is not None:
            policy_normalizer = policy_normalizer.to(obs.device)

        class DreamWaQNP3OOnnxWrapper(torch.nn.Module):
            def __init__(self, normalizer, cenet, actor_critic):
                super().__init__()
                self.normalizer = normalizer
                self.cenet = cenet
                self.actor_critic = actor_critic

            def forward(self, obs):
                if self.normalizer is not None:
                    obs = self.normalizer(obs)
                latent = self.cenet.encoder_inference(obs)
                obs_aug = torch.cat((obs, latent), dim=-1)
                return self.actor_critic.act_inference(obs_aug)

        model = DreamWaQNP3OOnnxWrapper(policy_normalizer, cenet, actor_critic)
        model.eval()

        os.makedirs(export_model_dir, exist_ok=True)
        export_path = os.path.join(export_model_dir, filename)
        torch.onnx.export(
            model,
            obs,
            export_path,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=11,
        )
        print(f"DreamWaQ-NP3O Policy exported to {export_path}")


class SymmetryDreamWaQNP3ORunnerV3(DreamWaQNP3ORunner):
    """DreamWaQ-NP3O v3 runner with online symmetry rollout in a unified frame."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.symmetry_cfg = train_cfg.get("symmetry", None)
        if self.symmetry_cfg is None or not self.symmetry_cfg.get("enabled", False):
            raise ValueError("SymmetryDreamWaQNP3ORunnerV3 requires symmetry.enabled=True in the runner config.")
        if self.env.num_envs % 2 != 0:
            raise ValueError("SymmetryDreamWaQNP3ORunnerV3 requires an even number of environments.")
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
        mirrored[self._mirror_start :] = self.symmetry_helper.mirror_group(group_name, mirrored[self._mirror_start :])
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
            "[INFO{}]: Initialization done, start learning (Constrained RL + symmetry).".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
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
                        for ri in range(self.env.num_rewards):
                            rframebuffer[ri].extend(rewards[dones < 1][:, ri].cpu().numpy().tolist())
                            rewbuffer[ri].extend(cur_reward_sum[new_ids][:, ri].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                critic_input = critic_obs if critic_obs is not None else obs
                self.alg.compute_returns(critic_input)
                self.alg.compute_cost_returns(critic_input)
                self.alg.current_learning_iteration = self.current_learning_iteration

                with torch.no_grad():
                    current_cost_returns = self.alg.storage.cost_returns.mean(dim=(0, 1))
                    self._mean_cost_return = current_cost_returns.detach().clone()
                    self._mean_discounted_cost_violation = (
                        (1.0 - self.alg.gamma) * current_cost_returns
                    ).detach().clone()

                    if hasattr(self.alg, "update_adaptive_constraints"):
                        self.alg.update_adaptive_constraints(current_cost_returns)
                        self.alg.storage.update_cost_violation(self.alg.gamma)

            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration += 1
            start = time.time()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def rollout_step(self, obs, critic_obs):
        num_single_obs = self.cenet.decoder.model[-1].out_features

        actions_policy, next_cenet_hidden_states = self.alg.act(obs, critic_obs, self.cenet_hidden_states)
        self.cenet_hidden_states = next_cenet_hidden_states.detach()
        actions_env = self._mirror_action_tail_to_env(actions_policy)

        next_obs, rewards, dones, infos = self.env.step(actions_env)
        costs = self._extract_costs(infos)

        env_ids_done = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids_done) > 0:
            self.cenet_hidden_states[:, env_ids_done] = 0.0

        next_critic_obs = infos["observations"].get("critic", None)
        next_obs, next_critic_obs, rewards, dones = (
            next_obs.to(self.device),
            next_critic_obs.to(self.device) if next_critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )

        infos["observations"] = self._mirror_observation_dict(infos["observations"])
        self._mirror_termination_observations(infos)

        next_obs = infos["observations"]["policy"]
        next_critic_obs = infos["observations"].get("critic", None)

        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                next_obs = normalizer(next_obs)
                infos["observations"]["policy"] = next_obs
            elif obs_group_name == "critic":
                if next_critic_obs is not None:
                    next_critic_obs = normalizer(next_critic_obs)
                    infos["observations"]["critic"] = next_critic_obs
            elif obs_group_name in infos["observations"]:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])

        if "termination_observations" in infos:
            term_obs = infos["termination_observations"]
            if isinstance(term_obs, dict):
                for k, v in term_obs.items():
                    if k in self.normalizers:
                        term_obs[k] = self.normalizers[k](v)
            else:
                if "critic" in self.normalizers:
                    infos["termination_observations"] = self.normalizers["critic"](term_obs)

        rewards_noClip = infos.get("rewards_noClip", rewards)
        if isinstance(rewards_noClip, torch.Tensor):
            rewards_noClip = rewards_noClip.to(self.device)

        self.alg.process_env_step(
            rewards=rewards,
            costs=costs,
            dones=dones,
            infos=infos,
            rewards_noClip=rewards_noClip,
            num_single_obs=num_single_obs,
            next_obs=next_obs,
            next_critic_obs=next_critic_obs,
        )

        if costs is not None and self.log_dir is not None:
            self._cost_buffer.extend(costs.sum(dim=-1).cpu().numpy().tolist())

        return next_obs, next_critic_obs, rewards, dones, infos
