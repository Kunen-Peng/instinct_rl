
from datetime import datetime
import torch
from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.modules import CENet

class DreamWaQRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        
        # Instantiate CENet
        # Assuming train_cfg["runner"]["cenet_class_name"] or similar exists, or hardcode/default
        # Defaulting to CENet class we imported
        if "cenet_class_name" in self.cfg:
             # If dynamic loading needed, we can do it, but direct instantiation implies we trust the class
             cenet_class_name = self.cfg["cenet_class_name"]
             # Assuming it refers to instinct_rl.modules.cenet.CENet which we imported
             # If it is a string like "CENet", we can use eval or globals if we imported it
             pass 
             
        # Initialize CENet
        # We need num_encoder_obs and num_single_obs
        # num_single_obs is likely num_obs (raw)
        # num_encoder_obs is likely frame_stack * num_single_obs
        
        # The env usually provides these? In DWL they are props of env.
        # base instinct_rl env might not have num_single_obs. We might need to config it.
        # Fallback: num_single_obs = env.num_obs // frame_stack ? 
        # Or look at policy cfg or env cfg.
        
        # In space_mjlab, num_obs is usually the full stacked obs if stacking is used.
        # But wait, DreamWaQ CENet takes history.
        
        # Let's assume the user configures "num_single_obs" in the runner config or env has it.
        # In DWL python code: self.env.num_encoder_obs, self.env.num_single_obs
        
        # We will try to get it from env, else cfg.
        num_single_obs = getattr(self.env, "num_single_obs", None)
        if num_single_obs is None:
             num_single_obs = self.cfg.get("num_single_obs", self.env.num_obs) # Fallback to full obs?
             
        num_encoder_obs = getattr(self.env, "num_encoder_obs", None)
        if num_encoder_obs is None:
             num_encoder_obs = self.cfg.get("num_encoder_obs", self.env.num_obs)
             
        self.cenet = CENet(num_encoder_obs, num_single_obs).to(self.device)
        
        # Re-initialize algorithm with CENet
        # super().__init__ initialized self.alg as PPO. We need PPODreamWaQ.
        # But wait, OnPolicyRunner uses class_name from config.
        # If config says "PPODreamWaQ", it will try to init it.
        # But PPODreamWaQ init needs `cenet` arg which generic OnPolicyRunner doesn't pass.
        # So independent of what super() did, we might need to re-init or patch.
        
        # Better approach: Override __init__ completely? Or let super fail? 
        # super() calls self.alg = alg_class(...)
        # If alg_class is PPODreamWaQ, it will fail due to missing `cenet`.
        
        # So we should probably override __init__ or part of the setup.
        # Since we can't easily partially override __init__, we have to copy-paste most of it or accept re-init cost.
        # But simple re-init means super() crashed.
        
        # Strategy: We assume the user configures "PPODreamWaQ" in config.
        # To avoid super() crash, we can hack: 
        # 1. Provide a dummy PPO class name to super, then replace self.alg.
        # 2. Or Copy-Paste OnPolicyRunner structure.
        
        # Copy-Paste seems safer to ensure correct wiring.
        pass

    # We will overwrite __init__ to inject CENet
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # CENet Init
        num_single_obs = getattr(self.env, "num_single_obs", self.cfg.get("num_single_obs"))
        num_encoder_obs = getattr(self.env, "num_encoder_obs", self.cfg.get("num_encoder_obs"))
        
        if num_single_obs is None or num_encoder_obs is None:
             raise ValueError("DreamWaQRunner requires num_single_obs and num_encoder_obs to be defined in Env or Runner Config")

        self.cenet = CENet(num_encoder_obs, num_single_obs).to(self.device)
        
        # Update obs_format to include estimator output
        # CENet output dim is est_z + est_v + est_h (default 16 + 3 + 0 = 19)
        # We can get it from self.cenet.encoder.model[-1].out_features if available or hardcode based on defaults
        est_dim = self.cenet.encoder.model[-1].out_features
        obs_format = env.get_obs_format()
        # Create a copy to avoid mutating original env property if reference (though get_obs_format usually returns new dict)
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
        single_obs = obs[..., -num_single_obs:]
        
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
        
        # Rewards No Clip
        # DWL uses a separate buffer. We'll use the returned rewards or info['rewards_noClip'] if available.
        rewards_noClip = infos.get("rewards_noClip", rewards) 
        if isinstance(rewards_noClip, torch.Tensor):
             rewards_noClip = rewards_noClip.to(self.device)
        
        # Process
        self.alg.process_env_step(rewards, dones, infos, rewards_noClip, single_obs, next_obs, next_critic_obs)
        
        return next_obs, next_critic_obs, rewards, dones, infos

    # Important: Need to support saving/loading CENet state
    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        # Add CENet state
        # PPODreamWaQ does NOT hold CENet state in its state_dict by default PPO logic?
        # PPO state_dict only saves actor_critic and optimizer.
        # We need to ensure PPODreamWaQ saves CENet or We save it here.
        # PPODreamWaQ has optimizer_cenet.
        
        # Let's check PPODreamWaQ implementation in instinct_rl/algorithms/dreamwaq.py
        # Current implementation inherits PPO state_dict which is: 
        # {"model_state_dict": ..., "optimizer_state_dict": ...}
        # It does NOT save cenet. 
        # We should modify PPODreamWaQ to override state_dict OR handle it in Runner.
        # Runner handle is easier if we don't want to touch algo more.
        
        run_state_dict["cenet_state_dict"] = self.cenet.state_dict()
        run_state_dict["optimizer_cenet_state_dict"] = self.alg.optimizer_cenet.state_dict()
        
        run_state_dict.update({
            "iter": self.current_learning_iteration,
            "infos": infos,
        })
        torch.save(run_state_dict, path)

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device) # weights_only=True? 
        self.alg.load_state_dict(loaded_dict)
        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
        if "optimizer_cenet_state_dict" in loaded_dict:
            self.alg.optimizer_cenet.load_state_dict(loaded_dict["optimizer_cenet_state_dict"])
            
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
            # encoder_inference usually returns the mean
            latent = self.cenet.encoder_inference(obs_norm)
            
            # 3. Concat (Augment observation)
            obs_aug = torch.cat((obs_norm, latent), dim=-1)
            
            # 4. Actor Inference
            return self.alg.actor_critic.act_inference(obs_aug)

        return policy

