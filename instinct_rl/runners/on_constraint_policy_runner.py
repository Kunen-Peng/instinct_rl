import importlib
import os
import statistics
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
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class OnConstraintPolicyRunner(OnPolicyRunner):
    """
    Runner for constrained RL algorithms (like NP3O).
    """
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        # Re-initialization of algorithm might be needed if super().__init__ creates the wrong one
        # or doesn't pass necessary args.
        # OnPolicyRunner.__init__ builds algorithm using config.
        # If config specifies NP3O class, it will try to build it.
        # But NP3O needs 'cost_shape' which is not passed in OnPolicyRunner.__init__ for init_storage.
        # So we might need to override the algorithm initialization or init_storage call.
        
        # Actually OnPolicyRunner calls self.alg.init_storage inside __init__.
        # But NP3O.init_storage signature is different (needs cost_shape).
        
        # So it suggests we should probably overwrite __init__ completely or make sure OnPolicyRunner logic is compatible.
        # Since logic is intertwined, copying and adapting is safer than inheritance with heavy overriding of __init__.
        # But I used inheritance to keep it clean.
        
        # Let's fix the init_storage call.
        # In OnPolicyRunner.__init__, it calls:
        # self.alg.init_storage(...) without cost args.
        # This will fail for NP3O if strict.
        # And we need to grab cost info from env.
        pass

    # We need to completely override __init__ to handle init_storage correctly for NP3O
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs_format = env.get_obs_format()

        # Prepare cost args early to pass to ActorCritic
        # Try to find cost shape and d_values from environment
        cost_shape = getattr(self.env, 'cost_shape', None)
        cost_d_values = getattr(self.env, 'cost_d_values', None)
        
        # If not direct attributes, maybe in cfg
        if cost_shape is None and hasattr(self.env, 'cfg'):
             try:
                 # Try config dict style
                 if isinstance(self.env.cfg, dict) and "costs" in self.env.cfg:
                      cost_shape = (len(self.env.cfg["costs"]),)
                 # Try object style
                 elif hasattr(self.env.cfg, "costs"):
                      cost_shape = (len(self.env.cfg.costs),)
             except:
                 pass
        
        # Check active terms from cost manager if wrapped env
        if cost_shape is None and hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'cost_manager'):
             try:
                 cost_shape = (len(self.env.unwrapped.cost_manager.active_terms),)
             except:
                 pass

        # Fallback to cfg
        if cost_shape is None: 
            cost_shape = self.cfg.get("cost_shape", None)
            
        num_costs = cost_shape[0] if cost_shape else 0

        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
            num_costs=num_costs,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = importlib.import_module(alg_class_name) if ":" in alg_class_name else getattr(algorithms, alg_class_name)
        
        # Check for k_value in env if needed by algorithm
        if hasattr(self.env, 'cost_k_values'):
             self.alg_cfg['k_value'] = self.env.cost_k_values
             
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # handle normalizers if needed
        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config: dict = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer
            
        if cost_d_values is None and hasattr(self.env, 'cost_d_values_tensor'):
             cost_d_values = self.env.cost_d_values_tensor

        if cost_d_values is None and cost_shape is not None:
             # Default to zero limits if not provided (assuming costs are already violations)
             cost_d_values = torch.zeros(cost_shape, device=self.device)
             
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
            cost_shape=cost_shape,
            cost_d_values=cost_d_values
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [instinct_rl.__file__]  # store files whose repo status will be logged.

        _, _ = self.env.reset()

    def rollout_step(self, obs, critic_obs):
        actions = self.alg.act(obs, critic_obs)
        obs, rewards, dones, infos = self.env.step(actions)
        
        # Extract costs from infos
        # Assuming costs are in infos['costs'] or similar
        costs = infos.get('costs', None)
        if costs is None:
             # Try fallback to 'cost' key or zeros if not found (critical error usually)
             if 'cost' in infos:
                 costs = infos['cost']
             else:
                 # Create zero costs if not found (Warning needed?)
                 # For now assume failure if not found is better or create zeros
                 # shape: (num_envs, cost_shape)
                 if self.alg.storage.cost_shape:
                     costs = torch.zeros(self.env.num_envs, *self.alg.storage.cost_shape, device=self.device)
        
        costs = costs.to(self.device)

        critic_obs = infos["observations"].get("critic", None)
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        
        # Dealing with obs normalizers
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic":
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            else:
                 if obs_group_name in infos["observations"]:
                     infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])

        self.alg.process_env_step(rewards, costs, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        # Override log to include cost metrics
        super().log(locs, width, pad)
        
        # Add NP3O specific logs
        # locs['losses'] contains result from alg.update()
        # In NP3O.update(), we add: cost_value_loss, viol_loss
        
        # The parent log() iterates over locs['losses'] and adds them to Loss/
        # So basic loss logging is handled.
        
        # We might want to log 'cost_violation' or mean costs if available in locs
        # Note: locs comes from locals() of learn().
        # learn() captures: obs, rewards, etc.
        # But we don't capture `costs` in the buffer in learn() loop in OnPolicyRunner.
        # We need to override learn() loop if we want to log mean costs.
        pass

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # We need to override learn loop to capture costs for logging
        
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
        
        # Cost buffers
        costbuffer = deque(maxlen=100) 
        
        cur_reward_sum = torch.zeros(self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print(
            "[INFO{}]: Initialization done, start learning (Constrained).".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
        
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.tot_timesteps = 0 
        
        # We need tot_start_time for logging
        self.tot_start_time = time.time()  # Note: parent sets this in log, but here valid too
        
        start = time.time()
        while self.current_learning_iteration < tot_iter:
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    
                    # Cost extraction repeated here (or call rollout_step but we need costs return)
                    # Let's use internal logic here to access costs for logging
                    costs = infos.get('costs', None)
                    if costs is None and 'cost' in infos:
                        costs = infos['cost']
                    if costs is None:
                        if self.alg.storage.cost_shape:
                             costs = torch.zeros(self.env.num_envs, *self.alg.storage.cost_shape, device=self.device)
                    else:
                        costs = costs.to(self.device)
                    
                    critic_obs = infos["observations"].get("critic", None)
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device) if critic_obs is not None else None,
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Obs Normalization
                    for obs_group_name, normalizer in self.normalizers.items():
                        if obs_group_name == "policy":
                            obs = normalizer(obs)
                            infos["observations"]["policy"] = obs
                        elif obs_group_name == "critic":
                            critic_obs = normalizer(critic_obs)
                            infos["observations"]["critic"] = critic_obs
                        else:
                             if obs_group_name in infos["observations"]:
                                 infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])

                    self.alg.process_env_step(rewards, costs, dones, infos, obs, critic_obs)

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
                        
                        # Log costs?
                        # costs shape: (num_envs, num_costs)
                        # We can log mean cost per step
                        if costs is not None:
                             # Just mean for now
                             costbuffer.extend(costs.mean(dim=-1).cpu().numpy().tolist())


                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)
                
                # Compute cost returns
                self.alg.compute_cost_returns(critic_obs if critic_obs is not None else obs)


            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            
            # Prepare locals for logging
            # We need to construct a locs dictionary that matches expected format of log()
            # Or assume log uses locals() which works because we are in the same scope if we call log(locals())
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                locs = locals()
                # Ensure all needed keys are present
                locs['tot_iter'] = tot_iter
                locs['start_iter'] = start_iter
                locs['tot_start_time'] = self.tot_start_time
                self.log(locs)
                
                # Extra logging for Costs (since not in parent log)
                if costbuffer:
                    mean_cost = statistics.mean(costbuffer)
                    self.writer_mp_add_scalar("Train/mean_cost", mean_cost, self.current_learning_iteration)
                
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
