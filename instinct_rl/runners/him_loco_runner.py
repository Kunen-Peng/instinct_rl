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
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class HIMLocoRunner:
    """Runner for HIMLoco training.
    
    This runner is designed to work with the HIMLocoVecEnvWrapper in space_mjlab.
    It expects the environment to return observation history suitable for HIMEstimator.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """
        Args:
            env: Vector environment (must be HIMLocoVecEnvWrapper compatible)
            train_cfg: Training configuration dict
            log_dir: Directory for logging (optional)
            device: Device to use for computation
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs_format = env.get_obs_format()

        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = importlib.import_module(alg_class_name.split(":")[0]) if ":" in alg_class_name else getattr(algorithms, alg_class_name)
        self.alg: algorithms.HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.use_termination_obs = self.cfg.get("use_termination_obs", False)

        # handle normalizers if needed
        self.normalizers = {}
        # Handle termination observation normalizer if enabled
        if self.use_termination_obs:
            term_obs_format = getattr(env, "get_termination_obs_format", lambda: None)()
            if term_obs_format is not None:
                for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
                    if obs_group_name in term_obs_format:
                        pass
        
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config: dict = config.copy()
            normalizer_class = config.pop("class_name")
            if normalizer_class is not None:
                normalizer = modules.build_normalizer(
                    input_shape=get_subobs_size(obs_format[obs_group_name]),
                    normalizer_class_name=normalizer_class,
                    normalizer_kwargs=config,
                )
                normalizer.to(self.device)
                self.normalizers[obs_group_name] = normalizer
        
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
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
        
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
            
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        start = time.time()
        
        while self.current_learning_iteration < tot_iter:
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
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

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)

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
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def rollout_step(self, obs, critic_obs):
        """Execute one rollout step with optimized observation handling."""
        actions = self.alg.act(obs, critic_obs)
        obs, rewards, dones, infos = self.env.step(actions)
        critic_obs = infos["observations"].get("critic", None)
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        
        # Apply observation normalizers
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic" and critic_obs is not None:
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            elif obs_group_name in infos["observations"]:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])
        
        # Handle termination observations for bootstrapping
        next_critic_obs_for_bootstrap = None
        if self.use_termination_obs and critic_obs is not None:
            termination_env_ids = infos.get("termination_env_ids", torch.tensor([], dtype=torch.int64, device=self.device))
            termination_obs = infos.get("termination_observations", {})
            
            if len(termination_env_ids) > 0 and len(termination_obs) > 0:
                next_critic_obs_for_bootstrap = critic_obs.clone().detach()
                term_critic_obs = termination_obs.get("critic", None)
                if term_critic_obs is not None:
                    term_critic_obs = term_critic_obs.to(self.device)
                    if "critic" in self.normalizers:
                        term_critic_obs = self.normalizers["critic"](term_critic_obs)
                    next_critic_obs_for_bootstrap[termination_env_ids] = term_critic_obs.clone().detach()
        
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs, next_critic_obs_for_bootstrap)
        return obs, critic_obs, rewards, dones, infos

    # Logging, saving, loading and helper methods same as HIMOnPolicyRunner...
    # (Including them for completeness as `him_loco_runner.py` is a standalone file)
    
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs["tot_start_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            all_keys = set()
            for ep_info in locs["ep_infos"]:
                all_keys.update(ep_info.keys())
            for key in all_keys:
                infotensor = []
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info: continue
                    if not isinstance(ep_info[key], torch.Tensor): ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0: ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor.append(ep_info[key].to(self.device))
                if not infotensor: continue
                infotensor = torch.cat(infotensor)
                value = self.gather_stat_values(infotensor, "mean")
                self.writer_mp_add_scalar((key if key.startswith("Episode") else "Episode/" + key), value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        if dist.is_initialized(): fps *= dist.get_world_size()

        if not self.is_mp_rank_other_process():
            print(f"Iteration {self.current_learning_iteration}: {fps} steps/s, mean reward: {statistics.mean([statistics.mean(buf) for buf in locs['rewbuffer']]):.2f}")
            print(ep_string)

    def save(self, path, infos=None):
        if self.is_mp_rank_other_process(): return
        run_state_dict = self.alg.state_dict()
        run_state_dict.update({f"{k}_normalizer_state_dict": v.state_dict() for k, v in self.normalizers.items()})
        run_state_dict.update({"iter": self.current_learning_iteration, "infos": infos})
        torch.save(run_state_dict, path)

    def load(self, path):
        if self.is_mp_rank_other_process(): return
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.load_state_dict(loaded_dict)
        for k, v in self.normalizers.items():
            if f"{k}_normalizer_state_dict" in loaded_dict:
                v.load_state_dict(loaded_dict[f"{k}_normalizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def is_mp_rank_other_process(self):
        return dist.is_initialized() and dist.get_rank() != 0
        
    def train_mode(self):
        self.alg.actor_critic.train()
        for normalizer in self.normalizers.values(): normalizer.train()

    def gather_stat_values(self, values: torch.Tensor, gather_op: str = "mean", remove_nan: bool = True):
        if remove_nan: values = values[~values.isnan()]
        values = values.to(self.device)
        if gather_op == "mean":
            num_values = torch.tensor([torch.numel(values)]).to(self.device)
            values = torch.sum(values)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.SUM)
                dist.all_reduce(num_values, dist.ReduceOp.SUM)
            values = values / num_values.item()
        return values

    def writer_mp_add_scalar(self, key, value, step):
        if not self.is_mp_rank_other_process():
            self.writer.add_scalar(key, value, step)
