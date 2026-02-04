"""
Constrained RL Runner for NP3O and similar algorithms.

This runner extends OnPolicyRunner to support constrained RL with minimal code duplication.
Key additions:
- Cost extraction from environment infos
- Cost returns computation
- Cost logging
"""
import os
import statistics
import importlib
import time
from collections import deque

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import instinct_rl
import instinct_rl.algorithms as algorithms
import instinct_rl.modules as modules
from instinct_rl.env import VecEnv
from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class OnConstraintPolicyRunner(OnPolicyRunner):
    """
    Runner for constrained RL algorithms (like NP3O).
    
    Extends OnPolicyRunner by:
    1. Building ActorCritic with cost critic
    2. Initializing storage with cost_shape and cost_d_values
    3. Extracting costs from environment step
    4. Computing cost returns alongside reward returns
    5. Logging cost metrics
    """
    
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # Don't call super().__init__() directly because we need to customize init_storage
        # Instead, replicate initialization with modifications
        
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs_format = env.get_obs_format()

        # Get cost information from environment
        cost_shape = self._get_cost_shape()
        cost_d_values = self._get_cost_d_values(cost_shape)
        num_costs = cost_shape[0] if cost_shape else 0

        # Build actor-critic with cost support
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
            num_costs=num_costs,
        ).to(self.device)

        # Build algorithm
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = (
            importlib.import_module(alg_class_name) 
            if ":" in alg_class_name 
            else getattr(algorithms, alg_class_name)
        )
        
        # Check for k_value in env if needed by algorithm
        if hasattr(self.env, 'cost_k_values'):
            self.alg_cfg['k_value'] = self.env.cost_k_values
             
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Handle normalizers
        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

        # Initialize storage with cost parameters
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [instinct_rl.__file__]
        
        # Cost tracking buffers
        self._cost_buffer = deque(maxlen=100)

        _, _ = self.env.reset()

    def _get_cost_shape(self):
        """Get cost shape from environment."""
        # Direct attribute (check value is not None)
        if hasattr(self.env, 'cost_shape') and self.env.cost_shape is not None:
            return self.env.cost_shape
            
        # From cost manager
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'cost_manager'):
            try:
                return (len(self.env.unwrapped.cost_manager.active_terms),)
            except:
                pass
                
        # From config
        if hasattr(self.env, 'cfg'):
            try:
                if isinstance(self.env.cfg, dict) and "costs" in self.env.cfg:
                    return (len(self.env.cfg["costs"]),)
                elif hasattr(self.env.cfg, "costs"):
                    return (len(self.env.cfg.costs),)
            except:
                pass
                
        # From runner config fallback
        return self.cfg.get("cost_shape", None)
    
    def _get_cost_d_values(self, cost_shape):
        """Get cost limit/threshold values from environment."""
        if hasattr(self.env, 'cost_d_values') and self.env.cost_d_values is not None:
            return self.env.cost_d_values
            
        # Default to zeros if not provided
        if cost_shape is not None:
            return torch.zeros(cost_shape, device=self.device)
        return None

    def _extract_costs(self, infos):
        """Extract costs from environment infos dict."""
        costs = infos.get('costs', None)
        if costs is None:
            costs = infos.get('cost', None)
        if costs is None:
            # Create zero costs as fallback
            if hasattr(self.alg, 'storage') and self.alg.storage.cost_shape:
                costs = torch.zeros(
                    self.env.num_envs, 
                    *self.alg.storage.cost_shape, 
                    device=self.device
                )
        else:
            costs = costs.to(self.device)
        return costs

    def rollout_step(self, obs, critic_obs):
        """
        Execute one rollout step with cost handling.
        
        Extends parent to:
        1. Extract costs from infos
        2. Pass costs to alg.process_env_step
        """
        actions = self.alg.act(obs, critic_obs)
        obs, rewards, dones, infos = self.env.step(actions)
        
        # Extract costs from infos
        costs = self._extract_costs(infos)
        
        # Process observations (same as parent)
        critic_obs = infos["observations"].get("critic", None)
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        
        # Apply normalizers
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic":
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            elif obs_group_name in infos["observations"]:
                infos["observations"][obs_group_name] = normalizer(
                    infos["observations"][obs_group_name]
                )
        
        # Call NP3O's process_env_step with costs
        # Note: NP3O.process_env_step has signature: (rewards, costs, dones, infos, next_obs, next_critic_obs)
        self.alg.process_env_step(rewards, costs, dones, infos, obs, critic_obs)
        
        # Track costs for logging
        if costs is not None and self.log_dir is not None:
            # Store mean cost per environment for logging
            self._cost_buffer.extend(costs.sum(dim=-1).cpu().numpy().tolist())
        
        return obs, critic_obs, rewards, dones, infos

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Main training loop.
        
        Extends parent to compute cost returns before update.
        """
        # Initialize distributed training
        if dist.is_initialized():
            self.alg.distributed_data_parallel()
            print(f"[INFO rank {dist.get_rank()}]: DistributedDataParallel enabled.")
            
        # Initialize tensorboard writer
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

        # Buffers for logging
        ep_infos = []
        step_infos = []
        rframebuffer = [deque(maxlen=2000) for _ in range(self.env.num_rewards)]
        rewbuffer = [deque(maxlen=100) for _ in range(self.env.num_rewards)]
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print(
            "[INFO{}]: Initialization done, start learning (Constrained RL).".format(
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
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
                    # Use rollout_step which handles cost extraction
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    
                    if len(rewards.shape) == 1:
                        rewards = rewards.unsqueeze(-1)

                    if self.log_dir is not None:
                        # Book keeping
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

                # Learning step
                start = stop
                critic_input = critic_obs if critic_obs is not None else obs
                
                # Compute returns for rewards (same as PPO)
                self.alg.compute_returns(critic_input)
                
                # --- Adaptive Threshold Update (Kim et al. 2024) ---
                # Update constraint thresholds based on current rollout performance
                # before computing cost returns/violation
                with torch.no_grad():
                    # Calculate mean cost per constraint from current rollout
                    # costs shape: [num_steps, num_envs, num_costs] -> mean: [num_costs]
                    current_mean_costs = self.alg.storage.costs.mean(dim=(0, 1))
                    
                    # Update adaptive thresholds (modifies storage.active_cost_d_values)
                    if hasattr(self.alg, 'update_adaptive_constraints'):
                        self.alg.update_adaptive_constraints(current_mean_costs)
                
                # Compute returns for costs (NP3O specific)
                # Now uses updated active thresholds for violation calculation
                self.alg.compute_cost_returns(critic_input)

            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
                
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
                
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration += 1
            start = time.time()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        """
        Log training metrics including cost-specific metrics.
        """
        # Call parent log for standard metrics
        super().log(locs, width, pad)
        
        # Log cost-specific metrics
        if self._cost_buffer:
            mean_cost = statistics.mean(self._cost_buffer)
            self.writer_mp_add_scalar("Train/mean_cost_per_step", mean_cost, self.current_learning_iteration)
            
            # Log k_value if available
            if hasattr(self.alg, 'k_value'):
                if isinstance(self.alg.k_value, torch.Tensor):
                    self.writer_mp_add_scalar(
                        "Train/k_value_mean", 
                        self.alg.k_value.mean().item(), 
                        self.current_learning_iteration
                    )
                else:
                    self.writer_mp_add_scalar(
                        "Train/k_value", 
                        self.alg.k_value, 
                        self.current_learning_iteration
                    )
            
            # Log adaptive thresholds
            if hasattr(self.alg, 'storage') and hasattr(self.alg.storage, 'active_cost_d_values'):
                active_d = self.alg.storage.active_cost_d_values
                target_d = self.alg.storage.target_cost_d_values
                if active_d is not None:
                    for i in range(len(active_d)):
                        self.writer_mp_add_scalar(
                            f'AdaptiveThreshold/cost_{i}_active',
                            active_d[i].item(),
                            self.current_learning_iteration
                        )
                        if target_d is not None:
                            self.writer_mp_add_scalar(
                                f'AdaptiveThreshold/cost_{i}_target',
                                target_d[i].item(),
                                self.current_learning_iteration
                            )
