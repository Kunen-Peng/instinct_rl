from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from instinct_rl.utils.utils import get_subobs_size
from .actor_critic import ActorCritic, get_activation
from .him_estimator import HIMEstimator


class HIMActorCritic(ActorCritic):
    """Hierarchical Imitation Mode (HIM) Actor-Critic network.
    
    Extends standard ActorCritic by using a HIM Estimator to extract
    velocity and latent features from observation history, enabling
    more sample-efficient learning from expert demonstrations.
    
    The estimator processes observation history and outputs:
    - Velocity estimation (3D)
    - Latent features (learned via contrastive learning)
    
    The policy network uses [current_obs, velocity, latent] as input.
    """

    is_recurrent = False

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
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
        **kwargs,
    ):
        """Initialize HIM Actor-Critic.
        
        Args:
            obs_format: Observation format dictionary
            num_actions: Number of actions
            actor_hidden_dims: Actor hidden dimensions
            critic_hidden_dims: Critic hidden dimensions
            activation: Activation function name
            init_noise_std: Initial noise standard deviation
            num_rewards: Number of reward outputs
            history_size: Number of observation history steps
            num_one_step_obs: Dimension of single observation (auto-computed if None)
            enc_hidden_dims: Estimator encoder hidden dimensions
            tar_hidden_dims: Estimator target encoder hidden dimensions
            num_prototype: Number of prototypes for contrastive learning
            temperature: Temperature for softmax in estimator
        """
        # Note: We don't call super().__init__() immediately because we need to set up
        # obs_format first based on HIM requirements
        
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

        # Verify observation dimensions
        if policy_obs_size != history_size * num_one_step_obs:
            print(
                f"[HIMActorCritic WARNING] Policy obs size ({policy_obs_size}) != "
                f"history_size ({history_size}) * num_one_step_obs ({num_one_step_obs})"
            )

        # Initialize HIM Estimator
        # The estimator expects flattened history from ObservationManager (oldest_first format)
        self.estimator = HIMEstimator(
            temporal_steps=history_size,
            num_one_step_obs=num_one_step_obs,
            enc_hidden_dims=enc_hidden_dims,
            tar_hidden_dims=tar_hidden_dims,
            activation=activation,
            num_prototype=num_prototype,
            temperature=temperature,
            history_format="oldest_first",  # Matches CircularBuffer output
        )

        # Actor input dimension: current_obs + velocity(3) + latent_features
        mlp_input_dim_a = num_one_step_obs + 3 + enc_hidden_dims[-1]
        
        # Critic input dimension (usually includes additional privileged observations)
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
        """Reset any internal states (for recurrent networks)."""
        pass

    def forward(self):
        raise NotImplementedError

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
        """Update action distribution based on observation history.
        
        Extracts velocity and latent features from observation history, then feeds them
        along with current observation to the actor network.
        
        The input obs_history is a flattened tensor from ObservationManager containing
        the full history in order: [obs_t0, obs_t1, ..., obs_t(H-1)] where t0 is oldest
        and t(H-1) is newest (when flatten_history_dim=True).
        
        Args:
            obs_history: Flattened observation history tensor [batch_size, history_size * num_one_step_obs]
        """
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        
        # Extract current observation from history
        # For oldest_first format: current obs is the last num_one_step_obs elements
        current_obs = obs_history[:, -self.num_one_step_obs:]
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history, **kwargs):
        """Sample action from the learned policy.
        
        Args:
            obs_history: Observation history tensor
        
        Returns:
            Sampled actions
        """
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Action tensor
        
        Returns:
            Log probability sum over action dimensions
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, **kwargs):
        """Deterministic action selection for inference.
        
        Returns the mean of the action distribution (no sampling).
        
        The input obs_history is a flattened tensor from ObservationManager containing
        the full history in order: [obs_t0, obs_t1, ..., obs_t(H-1)].
        
        Args:
            obs_history: Flattened observation history tensor
        
        Returns:
            Mean actions
        """
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        
        # Extract current observation from history (newest observation)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function.
        
        Args:
            critic_observations: Critic observation tensor(s)
        
        Returns:
            Value predictions
        """
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

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        """Clip action standard deviation."""
        self.std.copy_(self.std.clip(min=min, max=max))

    @property
    def obs_segments(self):
        """Observation segments for policy network."""
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        """Observation segments for critic network."""
        return self.__critic_obs_segments
    
    @property
    def obs_history_length(self):
        """Return the observation history length (temporal_steps).
        
        This property exposes the history length to runners and other modules
        that need to know how many steps of history are expected.
        """
        return self.history_size

    def export_as_onnx(self, observations, filedir):
        """Export actor and estimator to ONNX format."""
        self.eval()
        import os

        with torch.no_grad():
            # Export estimator (optional, mainly for documentation)
            torch.onnx.export(
                self.estimator,
                observations,
                os.path.join(filedir, "him_estimator.onnx"),
                input_names=["obs_history"],
                output_names=["velocity", "latent"],
                opset_version=12,
            )
            print(f"Exported HIM Estimator to {os.path.join(filedir, 'him_estimator.onnx')}")

            # Export full actor
            vel, latent = self.estimator(observations)
            # Current observation is the newest (last num_one_step_obs elements)
            current_obs = observations[:, -self.num_one_step_obs:]
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
