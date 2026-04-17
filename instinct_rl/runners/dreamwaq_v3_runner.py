import time
from collections import deque

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from instinct_rl.algorithms.dreamwaq_v3 import PPODreamWaQRecurrentV3, PPODreamWaQRecurrentV3SGMA
from instinct_rl.modules.cenet_v3 import CENetV3 as CENet
from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.utils.utils import store_code_state


class DreamWaQRecurrentRunnerV3(OnPolicyRunner):
    """DreamWaQ v3 runner.

    V3 keeps the V2 rollout/inference path, but swaps in the sequence-aware
    CENet and recurrent estimator update.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs", self.env.num_obs))
        if num_single_obs is None:
            raise ValueError("DreamWaQRecurrentRunnerV3 requires num_single_obs")

        cenet_cfg = {}
        if "cenet" in self.alg_cfg:
            cenet_cfg = self.alg_cfg.pop("cenet")
        elif "cenet" in self.cfg:
            cenet_cfg = self.cfg["cenet"]

        if "num_history_steps" not in cenet_cfg:
            if hasattr(self.env, "history_length"):
                cenet_cfg["num_history_steps"] = self.env.history_length
            elif hasattr(self.env.unwrapped, "history_length"):
                cenet_cfg["num_history_steps"] = self.env.unwrapped.history_length

        obs_format = env.get_obs_format()
        if cenet_cfg.get("num_history_steps", 1) > 1 and "term_dims" not in cenet_cfg:
            history_size = cenet_cfg["num_history_steps"]
            inferred_term_dims = []
            obs_segments = obs_format["policy"]
            for term_name, term_shape in obs_segments.items():
                if term_name == "estimator":
                    continue
                flat_dim = term_shape[0] if isinstance(term_shape, tuple) else term_shape
                if flat_dim % history_size != 0:
                    raise ValueError(f"Term '{term_name}' dim {flat_dim} not divisible by history_size {history_size}")
                inferred_term_dims.append(flat_dim // history_size)
            cenet_cfg["term_dims"] = inferred_term_dims
            if "obs_layout" not in cenet_cfg:
                cenet_cfg["obs_layout"] = "term_major"

        # Build CENet — pass RNN params only when encoder_type is RNN.
        encoder_type = cenet_cfg.get("encoder_type", "rnn")
        cenet_kwargs = dict(cenet_cfg)
        if encoder_type == "rnn":
            cenet_kwargs.setdefault("rnn_hidden_size", cenet_kwargs.pop("rnn_hidden_size", 256))
            cenet_kwargs.setdefault("rnn_num_layers", cenet_kwargs.pop("rnn_num_layers", 1))
            # rnn_type is not a V2/V3 param; V2 always uses GRU.
            cenet_kwargs.pop("rnn_type", None)
        else:
            # MLP mode — strip RNN-specific keys that may be in the config.
            cenet_kwargs.pop("rnn_type", None)
            cenet_kwargs.pop("rnn_hidden_size", None)
            cenet_kwargs.pop("rnn_num_layers", None)

        self.cenet = CENet(
            num_encoder_obs,
            num_single_obs,
            **cenet_kwargs,
        ).to(self.device)

        # Hidden states only needed for RNN encoder.
        if self.cenet.rnn is not None:
            self.cenet_hidden_states = torch.zeros(
                self.cenet.rnn.num_layers,
                self.env.num_envs,
                self.cenet.rnn.hidden_size,
                device=self.device,
            )
        else:
            self.cenet_hidden_states = None

        est_dim = self.cenet.latent_dim
        obs_format["policy"]["estimator"] = (est_dim,)

        import instinct_rl.modules as modules

        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)

        self.alg = PPODreamWaQRecurrentV3(actor_critic, self.cenet, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format,
            self.env.num_actions,
            self.env.num_rewards,
            num_single_obs=num_single_obs,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = []

        _, _ = self.env.reset()

        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config = config.copy()

            group_shape = obs_format[obs_group_name]
            if isinstance(group_shape, dict):
                input_shape = 0
                for k, v in group_shape.items():
                    if k == "estimator":
                        continue
                    input_shape += v[0] if isinstance(v, tuple) else v
            elif isinstance(group_shape, tuple):
                input_shape = group_shape[0]
            else:
                input_shape = group_shape

            normalizer = modules.build_normalizer(
                input_shape=input_shape,
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.cenet.to(device)

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)

        if self.cenet.rnn is not None:
            # --- RNN mode: stateful policy with hidden states ---
            class StatefulPolicy:
                def __init__(self, runner, device):
                    self.runner = runner
                    self.hidden_states = None
                    self.device = device

                def __call__(self, obs):
                    if self.hidden_states is None:
                        batch_size = obs.shape[0]
                        num_layers = self.runner.cenet.rnn.num_layers
                        hidden_size = self.runner.cenet.rnn.hidden_size
                        self.hidden_states = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)

                    if self.hidden_states.device != obs.device:
                        self.hidden_states = self.hidden_states.to(obs.device)
                        self.device = obs.device

                    if "policy" in self.runner.normalizers:
                        obs_norm = self.runner.normalizers["policy"](obs)
                    else:
                        obs_norm = obs

                    latent_mean, next_states = self.runner.cenet.encoder_inference_recurrent(obs_norm, self.hidden_states)
                    self.hidden_states = next_states
                    obs_aug = torch.cat((obs_norm, latent_mean), dim=-1)
                    return self.runner.alg.actor_critic.act_inference(obs_aug)

                def reset(self, dones=None):
                    if hasattr(self.runner.alg, "actor_critic") and hasattr(self.runner.alg.actor_critic, "reset"):
                        self.runner.alg.actor_critic.reset(dones)
                    if self.hidden_states is not None:
                        if dones is None:
                            self.hidden_states.zero_()
                        else:
                            self.hidden_states[:, dones, :] = 0.0

            return StatefulPolicy(self, device)
        else:
            # --- MLP mode: stateless policy ---
            runner = self

            def policy(obs):
                obs_norm = runner.normalizers["policy"](obs) if "policy" in runner.normalizers else obs
                latent = runner.cenet.encoder_inference(obs_norm)
                obs_aug = torch.cat((obs_norm, latent), dim=-1)
                return runner.alg.actor_critic.act_inference(obs_aug)

            return policy

    def export_as_onnx(self, obs, export_model_dir, filename="policy.onnx"):
        self.eval_mode()

        policy_normalizer = self.normalizers.get("policy")
        cenet = self.cenet
        actor_critic = self.alg.actor_critic

        batch_size = obs.shape[0]
        num_layers = cenet.rnn.num_layers
        hidden_size = cenet.rnn.hidden_size
        cenet_hidden_states = torch.zeros(num_layers, batch_size, hidden_size, device=obs.device)

        if policy_normalizer is not None:
            policy_normalizer = policy_normalizer.to(obs.device)

        class DreamWaQRecurrentV3OnnxWrapper(torch.nn.Module):
            def __init__(self, normalizer, cenet, actor_critic):
                super().__init__()
                self.normalizer = normalizer
                self.cenet = cenet
                self.actor_critic = actor_critic

            def forward(self, obs, cenet_hidden_states):
                if self.normalizer is not None:
                    obs_norm = self.normalizer(obs)
                else:
                    obs_norm = obs

                latent_mean, next_cenet_hidden_states = self.cenet.encoder_inference_recurrent(
                    obs_norm, cenet_hidden_states
                )
                obs_aug = torch.cat((obs_norm, latent_mean), dim=-1)
                actions = self.actor_critic.act_inference(obs_aug)
                return actions, next_cenet_hidden_states

        model = DreamWaQRecurrentV3OnnxWrapper(policy_normalizer, cenet, actor_critic)
        model.eval()

        import os

        os.makedirs(export_model_dir, exist_ok=True)
        export_path = os.path.join(export_model_dir, filename)

        torch.onnx.export(
            model,
            (obs, cenet_hidden_states),
            export_path,
            verbose=True,
            input_names=["obs", "cenet_hidden_states"],
            output_names=["actions", "next_cenet_hidden_states"],
            dynamic_axes={
                "obs": {0: "batch"},
                "cenet_hidden_states": {1: "batch"},
                "actions": {0: "batch"},
                "next_cenet_hidden_states": {1: "batch"},
            },
            opset_version=12,
        )
        print(f"DreamWaQ V3 recurrent policy exported to {export_path}")

    def _build_symmetry_helper(self, helper_class_name: str):
        module_name, class_name = helper_class_name.rsplit(":", 1)
        module = __import__(module_name, fromlist=[class_name])
        helper_cls = getattr(module, class_name)
        helper_kwargs = {}
        encoder = getattr(self.cenet, "encoder", self.cenet)
        if hasattr(encoder, "num_history_steps"):
            helper_kwargs["history_length"] = encoder.num_history_steps
        if hasattr(encoder, "obs_layout"):
            helper_kwargs["obs_layout"] = encoder.obs_layout
        return helper_cls(self.env.get_obs_format(), **helper_kwargs)

    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        run_state_dict["cenet_state_dict"] = self.cenet.state_dict()
        run_state_dict["optimizer_cenet_state_dict"] = self.alg.optimizer_cenet.state_dict()
        run_state_dict.update(
            {
                f"{group_name}_normalizer_state_dict": normalizer.state_dict()
                for group_name, normalizer in self.normalizers.items()
            }
        )
        run_state_dict.update(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        )
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
            self.current_learning_iteration += 1
            start = time.time()

        self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")

    def rollout_step(self, obs, critic_obs):
        num_single_obs = self.cenet.decoder.model[-1].out_features

        act_output = self.alg.act(obs, critic_obs, self.cenet_hidden_states)
        if isinstance(act_output, tuple):
            actions, next_cenet_hidden_states = act_output
            self.cenet_hidden_states = next_cenet_hidden_states.detach()
        else:
            actions = act_output

        next_obs, rewards, dones, infos = self.env.step(actions)

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
            else:
                if obs_group_name in infos["observations"]:
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

        self.alg.process_env_step(rewards, dones, infos, rewards_noClip, num_single_obs, next_obs, next_critic_obs)
        return next_obs, next_critic_obs, rewards, dones, infos


class SGMADreamWaQRecurrentRunnerV3(DreamWaQRecurrentRunnerV3):
    """DreamWaQ v3 runner with SGMA-style offline symmetry augmentation."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.symmetry_cfg = train_cfg.get("symmetry", None)
        if self.symmetry_cfg is None or not self.symmetry_cfg.get("enabled", False):
            raise ValueError("SGMADreamWaQRecurrentRunnerV3 requires symmetry.enabled=True in the runner config.")

        self.symmetry_helper = self._build_symmetry_helper(self.symmetry_cfg["helper_class_name"])

        alg_kwargs = self.alg_cfg.copy()
        alg_kwargs.pop("class_name", None)
        actor_critic = self.alg.actor_critic
        self.alg = PPODreamWaQRecurrentV3SGMA(
            actor_critic,
            self.cenet,
            device=self.device,
            symmetry_helper_class_name=self.symmetry_cfg["helper_class_name"],
            **alg_kwargs,
        )
        obs_format = self.env.get_obs_format()
        obs_format["policy"]["estimator"] = (self.cenet.latent_dim,)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format,
            self.env.num_actions,
            self.env.num_rewards,
            num_single_obs=getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs")),
        )
        self.alg.configure_sgma(
            self.env.get_obs_format(),
            self.symmetry_cfg["helper_class_name"],
            policy_normalizer=self.normalizers.get("policy"),
            critic_normalizer=self.normalizers.get("critic"),
        )


class SymmetryDreamWaQRecurrentRunnerV3(DreamWaQRecurrentRunnerV3):
    """DreamWaQ V3 runner with online symmetry rollout in a unified coordinate frame."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.symmetry_cfg = train_cfg.get("symmetry", None)
        if self.symmetry_cfg is None or not self.symmetry_cfg.get("enabled", False):
            raise ValueError("SymmetryDreamWaQRecurrentRunnerV3 requires symmetry.enabled=True in the runner config.")
        if self.env.num_envs % 2 != 0:
            raise ValueError("SymmetryDreamWaQRecurrentRunnerV3 requires an even number of environments.")
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
            self.current_learning_iteration += 1
            start = time.time()

        self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")

    def rollout_step(self, obs, critic_obs):
        num_single_obs = self.cenet.decoder.model[-1].out_features

        act_output = self.alg.act(obs, critic_obs, self.cenet_hidden_states)
        if isinstance(act_output, tuple):
            actions_policy, next_cenet_hidden_states = act_output
            self.cenet_hidden_states = next_cenet_hidden_states.detach()
        else:
            actions_policy = act_output
        actions_env = self._mirror_action_tail_to_env(actions_policy)

        next_obs, rewards, dones, infos = self.env.step(actions_env)

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
            else:
                if obs_group_name in infos["observations"]:
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

        self.alg.process_env_step(rewards, dones, infos, rewards_noClip, num_single_obs, next_obs, next_critic_obs)
        return next_obs, next_critic_obs, rewards, dones, infos
