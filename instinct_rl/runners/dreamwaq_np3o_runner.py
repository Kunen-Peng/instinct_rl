
import os
import torch
from instinct_rl.runners.on_constraint_policy_runner import OnConstraintPolicyRunner
from instinct_rl.modules import CENet
import instinct_rl.algorithms.dreamwaq_np3o as dreamwaq_np3o_alg
from instinct_rl.utils.utils import get_subobs_size

class DreamWaQNP3ORunner(OnConstraintPolicyRunner):
    """
    Runner for DreamWaQ-NP3O: Constrained RL + DreamWaQ Estimator.
    
    Combines logic from OnConstraintPolicyRunner (costs) and DreamWaQRunner (estimator).
    """
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # We perform valid initialization first before calling super/setup to ensure
        # we can inject the correct algorithm setup logic.
        
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # --- DreamWaQ Setup (CENet) ---
        # Try to get from Env (priority), then Config
        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs", self.env.num_obs))
        
        if num_single_obs is None:
             raise ValueError("DreamWaQNP3ORunner requires num_single_obs to be defined in Env or Runner Config")
        
        print(f"[DreamWaQNP3ORunner] Initialized CENet with num_encoder_obs={num_encoder_obs}, num_single_obs={num_single_obs}")

        # Initialize CENet
        self.cenet = CENet(num_encoder_obs, num_single_obs).to(self.device)
        
        # Start initializing components similar to OnConstraintPolicyRunner
        obs_format = env.get_obs_format()
        
        # Inject estimator dimension into obs_format for ActorCritic
        # CENet output dim is latent_dim (v+z+h)
        est_dim = self.cenet.latent_dim
        # Warning: ensure we are modifying a copy or updating correctly
        # Usually get_obs_format returns a new dict, but to be safe:
        if "policy" not in obs_format: obs_format["policy"] = {}
        obs_format["policy"]["estimator"] = (est_dim,)
        
        # Cost Setup (from OnConstraintPolicyRunner)
        cost_shape = self._get_cost_shape()
        cost_d_values = self._get_cost_d_values(cost_shape)
        num_costs = cost_shape[0] if cost_shape else 0

        # Build ActorCritic
        import instinct_rl.modules as modules
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
            num_costs=num_costs,
        ).to(self.device)
        
        # Initialize DreamWaQNP3O Algorithm
        # We skip dynamic import since we know the class
        alg_class = dreamwaq_np3o_alg.DreamWaQNP3O
        
        # Check for k_value in env if needed
        if hasattr(self.env, 'cost_k_values'):
            self.alg_cfg['k_value'] = self.env.cost_k_values
            
        self.alg = alg_class(
            actor_critic=actor_critic, 
            cenet=self.cenet, # Pass CENet here
            device=self.device, 
            **self.alg_cfg
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # Normalizers
        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config = config.copy()
            # Careful: if we are normalizing policy obs, the input shape might need check
            # Usually normalizer is applied on raw env obs BEFORE augmentation or split
            # For DreamWaQ, 'policy' obs from env is huge (history), we normalize that.
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]) - est_dim if obs_group_name=="policy" else get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

        # Init Storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
            num_single_obs=num_single_obs, # Extra arg for DreamWaQ storage
        )

        # Logging / Misc
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = []
        self._cost_buffer = [] # or deque in parent, we init fresh

        # Initial Reset
        _, _ = self.env.reset()

        # Cost tracking (re-init properly since we didn't call super().__init__)
        from collections import deque
        self._cost_buffer = deque(maxlen=100)

    def rollout_step(self, obs, critic_obs):
        """
        Rollout step incorporating:
        1. Action selection (with CENet estimation inside alg.act).
        2. Environment step.
        3. Cost extraction (NP3O).
        4. Normalization.
        5. Data processing (passing costs AND DreamWaQ extras).
        """
        # 1. & 2. Act
        # Note: alg.act expects 'obs' to be the raw policy input (without estimator)
        actions = self.alg.act(obs, critic_obs)
        
        # Step
        next_obs, rewards, dones, infos = self.env.step(actions)
        
        # 3. Cost Extraction
        costs = self._extract_costs(infos)
        
        # Prepare Tensors
        next_critic_obs = infos["observations"].get("critic", None)
        next_obs, next_critic_obs, rewards, dones = (
            next_obs.to(self.device),
            next_critic_obs.to(self.device) if next_critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        
        # 4. Normalizers
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                next_obs = normalizer(next_obs)
                infos["observations"]["policy"] = next_obs
            elif obs_group_name == "critic":
                if next_critic_obs is not None:
                    next_critic_obs = normalizer(next_critic_obs)
                    infos["observations"]["critic"] = next_critic_obs
            elif obs_group_name in infos["observations"]:
                infos["observations"][obs_group_name] = normalizer(
                    infos["observations"][obs_group_name]
                )

        # 5. Process Step
        # Get DreamWaQ specific extras
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
            next_critic_obs=next_critic_obs
        )
        
        # Logging costs
        if costs is not None and self.log_dir is not None:
            self._cost_buffer.extend(costs.sum(dim=-1).cpu().numpy().tolist())
            
        return next_obs, next_critic_obs, rewards, dones, infos

    def save(self, path, infos=None):
        """Save with CENet state."""
        run_state_dict = self.alg.state_dict()
        
        # Save CENet
        run_state_dict["cenet_state_dict"] = self.cenet.state_dict()
        run_state_dict["optimizer_cenet_state_dict"] = self.alg.optimizer_cenet.state_dict()
        
        run_state_dict.update({
            "iter": self.current_learning_iteration,
            "infos": infos,
        })
        torch.save(run_state_dict, path)

    def load(self, path):
        """Load with CENet state."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.load_state_dict(loaded_dict)
        
        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
        if "optimizer_cenet_state_dict" in loaded_dict:
            self.alg.optimizer_cenet.load_state_dict(loaded_dict["optimizer_cenet_state_dict"])
            
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Returns the inference policy with internal CENet handling."""
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
                # 1. Normalize
                if self.normalizer is not None:
                    obs = self.normalizer(obs)
                
                # 2. CENet Encoder
                latent = self.cenet.encoder_inference(obs)
                
                # 3. Concat
                obs_aug = torch.cat((obs, latent), dim=-1)
                
                # 4. Actor Inference
                return self.actor_critic.act_inference(obs_aug)

        model = DreamWaQNP3OOnnxWrapper(policy_normalizer, cenet, actor_critic)
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
            dynamic_axes={
                "obs": {0: "batch"},
                "actions": {0: "batch"}
            },
            opset_version=11
        )
        print(f"DreamWaQ-NP3O Policy exported to {export_path}")
