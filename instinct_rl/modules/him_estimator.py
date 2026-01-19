import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_activation(act_name):
    """Get activation function by name."""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    """Sinkhorn algorithm for optimal transport.
    
    Args:
        out: Score matrix
        eps: Temperature parameter
        iters: Number of iterations
    
    Returns:
        Normalized transport matrix
    """
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


class HIMEstimator(nn.Module):
    """Hierarchical Imitation Mode (HIM) Estimator module.
    
    This module estimates velocity and extracts latent features from observation history
    using contrastive learning with prototypes.
    
    NOTE: This estimator is designed to work with ObservationManager's history buffer.
    The input observation history should be a flattened tensor of shape:
    [batch_size, temporal_steps * num_one_step_obs]
    
    The history ordering convention is: [obs_t0, obs_t1, ..., obs_t(H-1)]
    where t0 is oldest and t(H-1) is newest. This matches CircularBuffer's ordering
    when flatten_history_dim=True.
    """

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=[128, 64, 16],
        tar_hidden_dims=[128, 64],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=32,
        temperature=3.0,
        history_format="oldest_first",
        **kwargs,
    ):
        """Initialize HIM Estimator.
        
        Args:
            temporal_steps: Number of temporal steps in history
            num_one_step_obs: Dimension of single observation step
            enc_hidden_dims: Encoder hidden layer dimensions
            tar_hidden_dims: Target encoder hidden layer dimensions
            activation: Activation function name
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            num_prototype: Number of prototypes for contrastive learning
            temperature: Temperature for softmax
            history_format: History ordering - "oldest_first" (default, from CircularBuffer)
                           or "newest_first" (reversed order)
        """
        if kwargs:
            print(
                "HIMEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.history_format = history_format
        
        if history_format not in ["oldest_first", "newest_first"]:
            raise ValueError(f"history_format must be 'oldest_first' or 'newest_first', got {history_format}")

        # Encoder: Takes full observation history (flattened)
        # Input shape: [batch_size, temporal_steps * num_one_step_obs]
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation_fn]
            enc_input_dim = enc_hidden_dims[l]
        # Output: velocity (3D) + latent features
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target: Processes current observation to get target latent
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation_fn]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype embedding for contrastive learning
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        """Get velocity and latent features from observation history.
        
        Returns detached tensors suitable for inference.
        """
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        """Forward pass for inference (detached output).
        
        Expects observation history as flattened tensor from ObservationManager.
        
        Args:
            obs_history: Flattened observation history [batch_size, temporal_steps * num_one_step_obs]
                        Ordering: oldest_first by default (from CircularBuffer)
        
        Returns:
            vel: Estimated velocity (3D)
            z: Normalized latent features
        """
        obs_input = self._prepare_obs_input(obs_history.detach())
        parts = self.encoder(obs_input)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        """Encode observation history to velocity and latent features.
        
        Expects observation history as flattened tensor from ObservationManager.
        
        Args:
            obs_history: Flattened observation history [batch_size, temporal_steps * num_one_step_obs]
                        Ordering: oldest_first by default (from CircularBuffer)
        
        Returns:
            vel: Estimated velocity (3D)
            z: Normalized latent features (not detached for training)
        """
        obs_input = self._prepare_obs_input(obs_history.detach())
        parts = self.encoder(obs_input)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z
    
    def _prepare_obs_input(self, obs_history):
        """Prepare observation history for network input.
        
        Handles history format conversion if needed (e.g., reversing order
        if network expects newest_first).
        
        Args:
            obs_history: Flattened observation history tensor
        
        Returns:
            Prepared observation input for encoder
        """
        if self.history_format == "oldest_first":
            # Default format from CircularBuffer - no conversion needed
            return obs_history
        elif self.history_format == "newest_first":
            # Reverse the history order
            batch_size = obs_history.shape[0]
            obs_reshaped = obs_history.reshape(batch_size, self.temporal_steps, self.num_one_step_obs)
            obs_reversed = torch.flip(obs_reshaped, dims=[1])
            return obs_reversed.reshape(batch_size, -1)
        else:
            raise ValueError(f"Unknown history_format: {self.history_format}")

    def update(self, obs_history, next_critic_obs, lr=None):
        """Update estimator using contrastive learning.
        
        Processes observation history from ObservationManager's CircularBuffer.
        
        Args:
            obs_history: Flattened observation history from ObservationManager
                        Shape: [batch_size, temporal_steps * num_one_step_obs]
                        Ordering: oldest_first (from CircularBuffer)
            next_critic_obs: Next critic observations (contains velocity estimate)
                           Expected format: [current_obs (num_one_step_obs), velocity (3), ...]
            lr: Optional learning rate override
        
        Returns:
            estimation_loss: Loss for velocity estimation
            swap_loss: Contrastive loss
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Extract velocity and current observation from next_critic_obs
        # Expected format: [current_obs (num_one_step_obs), velocity (3), ...]
        vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs.detach()[:, :self.num_one_step_obs]

        # Prepare observation history (handle format conversion if needed)
        z_s = self.encoder(self._prepare_obs_input(obs_history))
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        # Normalize latent features
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # Update prototypes (normalize)
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        # Compute scores
        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        # Compute targets using Sinkhorn
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        # Compute losses
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        # Backward pass
        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()
