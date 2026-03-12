
from datetime import datetime
import torch
from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.modules import CENet

class DreamWaQRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # CENet Init
        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs", self.env.num_obs))
        
        if num_single_obs is None:
             raise ValueError("DreamWaQRunner requires num_single_obs to be defined in Env or Runner Config")
             
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

        self.cenet = CENet(num_encoder_obs, num_single_obs, **cenet_cfg).to(self.device)
        
        # Update obs_format to include estimator output
        est_dim = self.cenet.latent_dim
        # obs_format = env.get_obs_format() # Already retrieved above
        obs_format["policy"]["estimator"] = (est_dim,)

        import instinct_rl.modules as modules
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)
        
        if self.cfg.get("load_cenet", False):
            # Load logic handled in load() usually
            pass

        import instinct_rl.algorithms as algorithms
        # Ensure we use PPODreamWaQ
        alg_class = algorithms.PPODreamWaQ
        
        self.alg = alg_class(actor_critic, self.cenet, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage with extra dims
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format,
            self.env.num_actions,
            self.env.num_rewards,
            num_single_obs=num_single_obs
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [] 

        _, _ = self.env.reset()
        
        # Normalizers
        self.normalizers = {}
        # ... (Same as base, omitted for brevity, add if needed)


    def rollout_step(self, obs, critic_obs):
        # We need to extract single_obs (last frame) from obs
        # obs shape: (num_envs, num_encoder_obs) where num_encoder_obs = history * single
        # single_obs = obs[..., -num_single_obs:]
        num_single_obs = self.cenet.decoder.model[-1].out_features         
        # Act
        actions = self.alg.act(obs, critic_obs)
        
        # Step
        next_obs, rewards, dones, infos = self.env.step(actions)
        
        # Deal with next_critic_obs
        next_critic_obs = infos["observations"].get("critic", None)
        
        next_obs, next_critic_obs, rewards, dones = (
            next_obs.to(self.device),
            next_critic_obs.to(self.device) if next_critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        
        # Apply Normalizers (Missing in previous implementation!)
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
                    
        # Apply normalization to termination_observations to avoid outlier explosion in loss
        if "termination_observations" in infos:
            term_obs = infos["termination_observations"]
            if isinstance(term_obs, dict):
                for k, v in term_obs.items():
                    if k in self.normalizers:
                        term_obs[k] = self.normalizers[k](v)
            else:
                # If term_obs is a direct tensor and normalizers have 'critic', assuming we are predicting critic
                if "critic" in self.normalizers:
                    infos["termination_observations"] = self.normalizers["critic"](term_obs)
        
        # Rewards No Clip
        # DWL uses a separate buffer. We'll use the returned rewards or info['rewards_noClip'] if available.
        rewards_noClip = infos.get("rewards_noClip", rewards) 
        if isinstance(rewards_noClip, torch.Tensor):
             rewards_noClip = rewards_noClip.to(self.device)
        
        # Process
        self.alg.process_env_step(rewards, dones, infos, rewards_noClip, num_single_obs, next_obs, next_critic_obs)
        
        return next_obs, next_critic_obs, rewards, dones, infos

    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        
        # Add CENet state
        run_state_dict["cenet_state_dict"] = self.cenet.state_dict()
        run_state_dict["optimizer_cenet_state_dict"] = self.alg.optimizer_cenet.state_dict()
        run_state_dict.update(
            {
                f"{group_name}_normalizer_state_dict": normalizer.state_dict()
                for group_name, normalizer in self.normalizers.items()
            }
        )
        
        run_state_dict.update({
            "iter": self.current_learning_iteration,
            "infos": infos,
        })
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
        """Returns the inference policy, handling CENet encoding and augmentation internally."""
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.cenet.to(device)

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)
            
        def policy(obs):
            # 1. Normalize
            if "policy" in self.normalizers:
                obs_norm = self.normalizers["policy"](obs)
            else:
                obs_norm = obs
            
            # 2. CENet Encoder (Use mean)
            latent = self.cenet.encoder_inference(obs_norm)
            
            # 3. Concat (Augment observation)
            obs_aug = torch.cat((obs_norm, latent), dim=-1)
            
            # 4. Actor Inference
            return self.alg.actor_critic.act_inference(obs_aug)

        return policy

    def export_as_onnx(self, obs, export_model_dir, filename="policy.onnx"):
        self.eval_mode()
        
        # Prepare components
        policy_normalizer = self.normalizers.get("policy")
        cenet = self.cenet
        actor_critic = self.alg.actor_critic
        
        if policy_normalizer is not None:
            policy_normalizer = policy_normalizer.to(obs.device)
            
        class DreamWaQOnnxWrapper(torch.nn.Module):
            def __init__(self, normalizer, cenet, actor_critic):
                super().__init__()
                self.normalizer = normalizer
                self.cenet = cenet
                self.actor_critic = actor_critic
                
            def forward(self, obs):
                # 1. Normalize
                if self.normalizer is not None:
                    obs = self.normalizer(obs)
                
                # Check for term_major reshaping needed before inference
                encoder = self.cenet.encoder
                if hasattr(encoder, "obs_layout") and encoder.obs_layout == "term_major" and encoder.term_dims is not None:
                    batch_size = obs.shape[0]
                    history_length = encoder.num_history_steps
                    
                    term_histories = []
                    offset = 0
                    for dim in encoder.term_dims:
                        chunk_size = dim * history_length
                        # Slice the flattened history
                        chunk = obs[:, offset:offset + chunk_size]
                        # Reshape to [batch, history, dim]
                        chunk = chunk.view(batch_size, history_length, dim)
                        term_histories.append(chunk)
                        offset += chunk_size
                        
                    # Concat along features
                    time_major = torch.cat(term_histories, dim=-1)
                    # Flatten back out to the input format MLPMixer expects initially
                    obs_reshaped = time_major.view(batch_size, -1)
                    latent = self.cenet.encoder_inference(obs_reshaped)
                else:
                    # 2. CENet Encoder
                    latent = self.cenet.encoder_inference(obs)
                
                # 3. Concat (Augment observation)
                obs_aug = torch.cat((obs, latent), dim=-1)
                
                # 4. Actor Inference
                return self.actor_critic.act_inference(obs_aug)

        model = DreamWaQOnnxWrapper(policy_normalizer, cenet, actor_critic)
        model.eval()
        
        import os
        os.makedirs(export_model_dir, exist_ok=True)
        export_path = os.path.join(export_model_dir, filename)
        
        torch.onnx.export(
            model,
            obs,
            export_path,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            # dynamic_axes={
            #     "obs": {0: "batch"},
            #     "actions": {0: "batch"}
            # },
            opset_version=12
        )
        print(f"DreamWaQ Policy exported to {export_path}")

class DreamWaQRecurrentRunner(DreamWaQRunner):
    """
    Runner for DreamWaQ with Recurrent CENet.
    Manages CENet hidden states during rollout and inference.
    """
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # CENet Init with Recurrent Args
        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs", self.env.num_obs))
        
        if num_single_obs is None:
             raise ValueError("DreamWaQRecurrentRunner requires num_single_obs")

        print(f"[DreamWaQRecurrentRunner] Initialized CENet with num_encoder_obs={num_encoder_obs}, num_single_obs={num_single_obs}")

        # Extract RNN args from config
        # Check if 'cenet' is in alg_cfg (preferred location) or top level
        cenet_cfg = {}
        if "cenet" in self.alg_cfg:
            cenet_cfg = self.alg_cfg.pop("cenet") # Remove to avoid collision in alg init
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
            rnn_type=cenet_cfg.pop("rnn_type", "gru"), # Default to GRU if using Recurrent Runner
            rnn_hidden_size=cenet_cfg.pop("rnn_hidden_size", 256),
            rnn_num_layers=cenet_cfg.pop("rnn_num_layers", 1),
            **cenet_cfg
        ).to(self.device)
        
        # State Tracking
        # Need to init hidden states for each env
        if self.cenet.rnn is not None:
            self.cenet_hidden_states = torch.zeros(
                self.cenet.rnn.num_layers, self.env.num_envs, self.cenet.rnn.rnn.hidden_size, device=self.device
            )
        else:
            raise ValueError("DreamWaQRecurrentRunner expects a recurrent CENet.")

        # Update obs_format
        est_dim = self.cenet.latent_dim
        # obs_format = env.get_obs_format() # Already retrieved above
        obs_format["policy"]["estimator"] = (est_dim,)

        import instinct_rl.modules as modules
        # Use StatefulEncoderActorCriticRecurrent if needed? 
        # But DreamWaQ passes CENet separately. Arguments to build_actor_critic are standard.
        # If backbone is recurrent, it's fine.
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)
        
        import instinct_rl.algorithms as algorithms
        # Use PPODreamWaQRecurrent
        alg_class = algorithms.PPODreamWaQRecurrent
        
        self.alg = alg_class(actor_critic, self.cenet, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format,
            self.env.num_actions,
            self.env.num_rewards,
            num_single_obs=num_single_obs
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
        # (Assuming normalizers init same as Parent, omitted for brevity or need to copy-paste if not calling super)
        # Copy-paste normalizer init for safety/completeness
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config = config.copy()
            
            group_shape = obs_format[obs_group_name]
            if isinstance(group_shape, dict):
                # Sum dimensions of all terms. Each value is typically a tuple (dim,).
                # Handle both tuple (dim,) and integer (though usually tuple in isaaclab)
                input_shape = 0
                for k, v in group_shape.items():
                    if k == "estimator":
                        continue
                    if isinstance(v, tuple):
                        input_shape += v[0]
                    else:
                        input_shape += v
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


    def rollout_step(self, obs, critic_obs):
        num_single_obs = self.cenet.decoder.model[-1].out_features         
        
        # Act with Recurrent State
        # self.alg.act is PPODreamWaQRecurrent.act
        # pass current cenet_hidden_states
        actions, next_cenet_hidden_states = self.alg.act(obs, critic_obs, self.cenet_hidden_states)
        
        self.cenet_hidden_states = next_cenet_hidden_states.detach()

        # Step Env
        next_obs, rewards, dones, infos = self.env.step(actions)
        
        # Reset hidden states for done envs
        # next_cenet_hidden_states shape: (num_layers, num_envs, hidden)
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
        
        # Normalizers
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
                    
        # Apply normalization to termination_observations to avoid outlier explosion in loss
        if "termination_observations" in infos:
            term_obs = infos["termination_observations"]
            if isinstance(term_obs, dict):
                for k, v in term_obs.items():
                    if k in self.normalizers:
                        term_obs[k] = self.normalizers[k](v)
            else:
                # If term_obs is a direct tensor and normalizers have 'critic', assuming we are predicting critic
                if "critic" in self.normalizers:
                    infos["termination_observations"] = self.normalizers["critic"](term_obs)
        
        rewards_noClip = infos.get("rewards_noClip", rewards) 
        if isinstance(rewards_noClip, torch.Tensor):
             rewards_noClip = rewards_noClip.to(self.device)
        
        self.alg.process_env_step(rewards, dones, infos, rewards_noClip, num_single_obs, next_obs, next_critic_obs)
        
        return next_obs, next_critic_obs, rewards, dones, infos

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.cenet.to(device)
            if self.cenet.rnn is not None:
                 pass

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)
            
        # Closure state
        
        class StatefulPolicy:
            def __init__(self, runner, device):
                self.runner = runner
                self.hidden_states = None
                self.device = device
                
            def __call__(self, obs):
                # Init states if needed
                if self.hidden_states is None:
                    batch_size = obs.shape[0]
                    num_layers = self.runner.cenet.rnn.num_layers
                    hidden_size = self.runner.cenet.rnn.rnn.hidden_size
                    self.hidden_states = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)
                
                # Check device consistency
                if self.hidden_states.device != obs.device:
                    self.hidden_states = self.hidden_states.to(obs.device)
                    self.device = obs.device

                # Normalize
                if "policy" in self.runner.normalizers:
                    obs_norm = self.runner.normalizers["policy"](obs)
                else:
                    obs_norm = obs
                
                # CENet Recurrent Inference
                # Use our new explicit method or manual call
                v_mean, next_states = self.runner.cenet.encoder_inference_recurrent(obs_norm, self.hidden_states)
                self.hidden_states = next_states
                
                # Actor Inference
                obs_aug = torch.cat((obs_norm, v_mean), dim=-1)
                
                return self.runner.alg.actor_critic.act_inference(obs_aug)
                
            def reset(self, dones=None):
                if self.hidden_states is not None:
                     if dones is None:
                         self.hidden_states.zero_()
                     else:
                         self.hidden_states[:, dones, :] = 0.0

        return StatefulPolicy(self, device)

    def export_as_onnx(self, obs, export_model_dir, filename="policy.onnx"):
        self.eval_mode()
        
        # Prepare components
        policy_normalizer = self.normalizers.get("policy")
        cenet = self.cenet
        actor_critic = self.alg.actor_critic
        
        # Initial hidden state for dummy input
        # obs shape: (batch, dim)
        batch_size = obs.shape[0]
        num_layers = cenet.rnn.num_layers
        hidden_size = cenet.rnn.rnn.hidden_size
        cenet_hidden_states = torch.zeros(num_layers, batch_size, hidden_size, device=obs.device)

        if policy_normalizer is not None:
            policy_normalizer = policy_normalizer.to(obs.device)
            
        class DreamWaQRecurrentOnnxWrapper(torch.nn.Module):
            def __init__(self, normalizer, cenet, actor_critic):
                super().__init__()
                self.normalizer = normalizer
                self.cenet = cenet
                self.actor_critic = actor_critic
                
            def forward(self, obs, cenet_hidden_states):
                # 1. Normalize
                if self.normalizer is not None:
                    obs_norm = self.normalizer(obs)
                else:
                    obs_norm = obs
                
                # Check for term_major reshaping needed before inference
                encoder = self.cenet.encoder
                if hasattr(encoder, "obs_layout") and encoder.obs_layout == "term_major" and encoder.term_dims is not None:
                    batch_size = obs_norm.shape[0]
                    history_length = encoder.num_history_steps
                    
                    term_histories = []
                    offset = 0
                    for dim in encoder.term_dims:
                        chunk_size = dim * history_length
                        # Slice the flattened history
                        chunk = obs_norm[:, offset:offset + chunk_size]
                        # Reshape to [batch, history, dim]
                        chunk = chunk.view(batch_size, history_length, dim)
                        term_histories.append(chunk)
                        offset += chunk_size
                        
                    # Concat along features
                    time_major = torch.cat(term_histories, dim=-1)
                    # Flatten back out to the input format MLPMixer expects initially
                    obs_reshaped = time_major.view(batch_size, -1)
                    v_mean, next_cenet_hidden_states = self.cenet.encoder_inference_recurrent(obs_reshaped, cenet_hidden_states)
                else:
                    # 2. CENet Recurrent Inference
                    # encoder_inference_recurrent returns (v_mean, next_hidden_states)
                    v_mean, next_cenet_hidden_states = self.cenet.encoder_inference_recurrent(obs_norm, cenet_hidden_states)
                
                # 3. Concat (Augment observation)
                # Ensure the original normalized obs is concatenated with the returned latent representation
                obs_aug = torch.cat((obs_norm, v_mean), dim=-1)
                
                # 4. Actor Inference
                actions = self.actor_critic.act_inference(obs_aug)
                
                return actions, next_cenet_hidden_states

        model = DreamWaQRecurrentOnnxWrapper(policy_normalizer, cenet, actor_critic)
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
                "cenet_hidden_states": {1: "batch"}, # (layers, batch, hidden)
                "actions": {0: "batch"},
                "next_cenet_hidden_states": {1: "batch"}
            },
            opset_version=12
        )
        print(f"DreamWaQ Recurrent Policy exported to {export_path}")

