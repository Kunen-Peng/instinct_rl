import torch

from instinct_rl.algorithms.dreamwaq_v3 import PPODreamWaQRecurrentV3
from instinct_rl.modules.cenet_v3 import CENetV3 as CENet
from instinct_rl.runners.dreamwaq_v2_runner import DreamWaQRecurrentRunnerV2


class DreamWaQRecurrentRunnerV3(DreamWaQRecurrentRunnerV2):
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

        self.cenet = CENet(
            num_encoder_obs,
            num_single_obs,
            rnn_type=cenet_cfg.pop("rnn_type", "gru"),
            rnn_hidden_size=cenet_cfg.pop("rnn_hidden_size", 256),
            rnn_num_layers=cenet_cfg.pop("rnn_num_layers", 1),
            **cenet_cfg,
        ).to(self.device)

        self.cenet_hidden_states = torch.zeros(
            self.cenet.rnn.num_layers,
            self.env.num_envs,
            self.cenet.rnn.hidden_size,
            device=self.device,
        )

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
