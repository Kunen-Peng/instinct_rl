from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from instinct_rl.utils.utils import get_subobs_size
from .actor_critic import ActorCritic, get_activation
from .him_estimator import HIMEstimator


def term_major_to_time_major(
    obs_history: torch.Tensor, 
    term_dims: List[int], 
    history_length: int
) -> torch.Tensor:
    """Convert term-major observation history to time-major format.
    
    Term-major layout (mjlab default):
        [A_t0, A_t1, ..., A_tH-1, B_t0, B_t1, ..., B_tH-1, ...]
        All history of term A, then all history of term B, etc.
        
    Time-major layout (HIM expected):
        [obs_t0, obs_t1, ..., obs_tH-1]
        where obs_ti = concat(A_ti, B_ti, ...)
        
    Args:
        obs_history: [batch_size, total_dim] where total_dim = sum(term_dims) * history_length
        term_dims: List of dimensions for each observation term
        history_length: Number of history timesteps
        
    Returns:
        Tensor of shape [batch_size, history_length * sum(term_dims)] in time-major order
    """
    batch_size = obs_history.shape[0]
    num_one_step_obs = sum(term_dims)
    
    # Split by term, each with shape [batch, term_dim * history_length]
    term_chunks = torch.split(
        obs_history, 
        [dim * history_length for dim in term_dims], 
        dim=-1
    )
    
    # Reshape each term to [batch, history_length, term_dim]
    term_histories = [
        chunk.reshape(batch_size, history_length, dim) 
        for chunk, dim in zip(term_chunks, term_dims)
    ]
    
    # Concatenate along term dimension: [batch, history_length, num_one_step_obs]
    time_major = torch.cat(term_histories, dim=-1)
    
    # Flatten back: [batch, history_length * num_one_step_obs]
    return time_major.reshape(batch_size, -1)


class HIMActorCritic(ActorCritic):
    """Hierarchical Imitation Mode (HIM) Actor-Critic network.
    
    Reference: HIMLoco/rsl_rl/modules/him_actor_critic.py
    
    Supports both time-major and term-major observation layouts:
    - time_major: [obs_t0, obs_t1, ..., obs_tH-1] (HIMLoco default)
    - term_major: [A_t0..A_tH-1, B_t0..B_tH-1, ...] (mjlab default)
    """

    is_recurrent = False

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions: int,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        # HIM-specific parameters
        history_size=10,
        num_one_step_obs=None,
        enc_hidden_dims=[128, 64, 16],
        tar_hidden_dims=[128, 64],
        num_prototype=32,
        temperature=3.0,
        # Observation layout parameters
        obs_layout: str = "time_major",  # "time_major" or "term_major"
        term_dims: Optional[List[int]] = None,  # Required if obs_layout="term_major"
        **kwargs,
    ):
        
        nn.Module.__init__(self)  # Initialize nn.Module directly

        # Get policy observation format
        self.__obs_format = obs_format
        self.__obs_segments = obs_format["policy"]
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])

        # Calculate observation dimensions
        policy_obs_size = get_subobs_size(self.__obs_segments)
        
        # If num_one_step_obs is not provided, infer from history_size
        if num_one_step_obs is None:
            num_one_step_obs = policy_obs_size // history_size
            print(f"[HIMActorCritic] Auto-computed num_one_step_obs: {num_one_step_obs}")
        
        self.history_size = history_size
        self.num_one_step_obs = num_one_step_obs
        self.num_actions = num_actions
        
        # Observation layout handling
        self.obs_layout = obs_layout
        if obs_layout == "term_major":
            # Auto-infer term_dims from obs_segments if not provided
            if term_dims is None:
                # Each segment value is a tuple like (dim,) or (history, dim)
                # For term-major with history, each term's flattened size = dim * history_length
                # But obs_segments already reflects the flattened size per term
                # We need the *single-step* dimension for each term
                inferred_term_dims = []
                for term_name, term_shape in self.__obs_segments.items():
                    # term_shape is like (flattened_dim,) which equals (single_dim * history_length,)
                    flat_dim = term_shape[0]
                    single_dim = flat_dim // history_size
                    if flat_dim % history_size != 0:
                        raise ValueError(
                            f"Term '{term_name}' has dim {flat_dim} which is not divisible by history_size={history_size}"
                        )
                    inferred_term_dims.append(single_dim)
                term_dims = inferred_term_dims
                print(f"[HIMActorCritic] Auto-inferred term_dims from obs_segments: {term_dims}")
            self.term_dims = term_dims
            # Validate
            expected_total = sum(term_dims) * history_size
            if expected_total != policy_obs_size:
                raise ValueError(
                    f"term_dims sum * history_size ({expected_total}) != policy_obs_size ({policy_obs_size})"
                )
            print(f"[HIMActorCritic] Using term-major layout with term_dims={term_dims}")
        else:
            self.term_dims = None
            print(f"[HIMActorCritic] Using time-major layout (no conversion needed)")

        # Initialize HIM Estimator
        self.estimator = HIMEstimator(
            temporal_steps=history_size,
            num_one_step_obs=num_one_step_obs,
            enc_hidden_dims=enc_hidden_dims,
            tar_hidden_dims=tar_hidden_dims,
            activation=activation,
            num_prototype=num_prototype,
            temperature=temperature,
            history_format="oldest_first",
        )

        # Actor input dimension: current_obs + velocity(3) + latent_features
        mlp_input_dim_a = num_one_step_obs + 3 + enc_hidden_dims[-1]
        
        # Critic input dimension
        mlp_input_dim_c = get_subobs_size(self.__critic_obs_segments)

        self.activation = activation
        self.mu_activation = kwargs.get("mu_activation", None)
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.mlp_input_dim_a = mlp_input_dim_a
        self.mlp_input_dim_c = mlp_input_dim_c

        # Build actor and critic networks
        self.actor = self._build_actor(num_actions)
        
        # Support multi-reward critics
        if num_rewards > 1:
            critic = self._build_critic(1)
            self.critics = nn.ModuleList([critic] + [self._build_critic(1) for _ in range(num_rewards - 1)])
        else:
            self.critic = self._build_critic(1)

        print(f"HIM Actor MLP: {self.actor}")
        print(f"HIM Estimator: {self.estimator.encoder}")
        if num_rewards > 1:
            print(f"HIM Multiple Critics: {len(self.critics)} in total")
        else:
            print(f"HIM Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _build_actor(self, num_actions):
        """Build actor network with HIM features."""
        activation_fn = get_activation(self.activation)
        actor_layers = []
        actor_layers.append(nn.Linear(self.mlp_input_dim_a, self.actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(self.actor_hidden_dims)):
            if l == len(self.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], num_actions))
                if self.mu_activation:
                    actor_layers.append(get_activation(self.mu_activation))
            else:
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], self.actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        return nn.Sequential(*actor_layers)

    def _build_critic(self, num_values=1):
        """Build critic network."""
        activation_fn = get_activation(self.activation)
        critic_layers = []
        critic_layers.append(nn.Linear(self.mlp_input_dim_c, self.critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(self.critic_hidden_dims)):
            if l == len(self.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], num_values))
            else:
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], self.critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        return nn.Sequential(*critic_layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def _convert_obs_layout(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Convert observation history to time-major format if needed.
        
        Args:
            obs_history: [batch_size, total_dim] observation history
            
        Returns:
            Observation history in time-major format for HIM processing
        """
        if self.obs_layout == "term_major" and self.term_dims is not None:
            return term_major_to_time_major(obs_history, self.term_dims, self.history_size)
        return obs_history

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        # Convert to time-major if needed
        obs_history_tm = self._convert_obs_layout(obs_history)
        
        with torch.no_grad():
            vel, latent = self.estimator(obs_history_tm)
        
        # Extract current observation from history
        # For oldest_first format: current obs is the last num_one_step_obs elements
        current_obs = obs_history_tm[:, -self.num_one_step_obs:]
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, **kwargs):
        # Convert to time-major if needed
        obs_history_tm = self._convert_obs_layout(obs_history)
        
        with torch.no_grad():
            vel, latent = self.estimator(obs_history_tm)
        
        current_obs = obs_history_tm[:, -self.num_one_step_obs:]
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if hasattr(self, "critics") and isinstance(critic_observations, list):
            value = torch.cat(
                [critic(critic_obs) for critic, critic_obs in zip(self.critics, critic_observations)],
                dim=-1,
            )
        elif hasattr(self, "critics"):
            value = torch.cat(
                [critic(critic_observations) for critic in self.critics],
                dim=-1,
            )
        else:
            value = self.critic(critic_observations)
        return value
    
    def update_estimator(self, obs_history, next_critic_obs, lr=None):
        """Update the HIM estimator with layout conversion.
        
        This method should be used instead of directly calling estimator.update()
        as it applies the necessary observation layout conversion.
        
        Args:
            obs_history: Observation history in environment format (may be term-major)
            next_critic_obs: Next critic observations for contrastive learning
            lr: Learning rate (optional)
            
        Returns:
            Tuple of (estimation_loss, swap_loss)
        """
        # Convert observation history to time-major format if needed
        obs_history_tm = self._convert_obs_layout(obs_history)
        
        return self.estimator.update(obs_history_tm, next_critic_obs, lr=lr)

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))

    @property
    def obs_segments(self):
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments
    
    @property
    def obs_history_length(self):
        return self.history_size

    def export_as_onnx(self, observations, filedir, input_transform=None):
        """Export actor and estimator to ONNX format."""
        self.eval()
        import os

        estimator_export = _EstimatorExportWrapper(self.estimator, input_transform)
        estimator_export.eval()

        processed_obs = input_transform(observations) if input_transform is not None else observations
        with torch.no_grad():
            # Export full actor
            vel, latent = self.estimator(processed_obs)
            current_obs = processed_obs[:, -self.num_one_step_obs:]
            actor_input = torch.cat((current_obs, vel, latent), dim=-1)
            
            torch.onnx.export(
                self.actor,
                actor_input,
                os.path.join(filedir, "actor.onnx"),
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
            print(f"Exported HIM Actor to {os.path.join(filedir, 'actor.onnx')}")


class _EstimatorExportWrapper(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module, input_transform: torch.nn.Module):
        super().__init__()
        self.estimator = estimator
        self.input_transform = input_transform

    def forward(self, observations):
        if self.input_transform is not None:
            observations = self.input_transform(observations)
        return self.estimator(observations)
