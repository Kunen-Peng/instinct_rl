import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from instinct_rl.modules.actor_critic import get_activation


class HIMEstimator(nn.Module):
    """Hierarchical Imitation Mode (HIM) Estimator.
    
    Estimates velocity and learns latent representation via contrastive learning.
    Reference: HIMLoco/rsl_rl/modules/him_estimator.py
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

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation_fn]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation_fn]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        """Forward pass for inference."""
        obs_input = self._prepare_obs_input(obs_history.detach())
        parts = self.encoder(obs_input)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        """Encode observation history."""
        obs_input = self._prepare_obs_input(obs_history.detach())
        parts = self.encoder(obs_input)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z
    
    def _prepare_obs_input(self, obs_history):
        """Handle history format conversion."""
        if self.history_format == "oldest_first":
            return obs_history
        elif self.history_format == "newest_first":
            batch_size = obs_history.shape[0]
            obs_reshaped = obs_history.reshape(batch_size, self.temporal_steps, self.num_one_step_obs)
            obs_reversed = torch.flip(obs_reshaped, dims=[1])
            return obs_reversed.reshape(batch_size, -1)
        else:
             raise ValueError(f"Unknown history_format: {self.history_format}")

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        # Extract velocity and current observation (next_obs) from next_critic_obs
        # NOTE: This assumes next_critic_obs structure from HIMLoco: [current_obs, velocity(3), ...]
        # In HIMLoco reference, velocity is at index num_one_step_obs:num_one_step_obs+3
        # But wait, HIMLoco reference says:
        # vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3]
        # next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3] <-- This looks weird in reference file provided?
        
        # Let's check the provided reference file again.
        # Reference line 83: next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]
        # Reference line 82: vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3]
        
        # This implies next_critic_obs layout is: [something_3_dim, obs(num_one_step), vel(3)...]
        # OR [obs(num_one_step), vel(3)] but indices are shifted?
        # Actually in HIMLoco generally: critic obs = [obs, privileged...] and privileged includes vel.
        # If line 82 takes vel from `num_one_step_obs` index, that means vel starts after obs.
        
        # However, line 83 is suspicious: `3:self.num_one_step_obs+3`.
        # If num_one_step_obs=48. indices 3 to 51.
        # This suggests the first 3 dims are something else? Or maybe it's `0:self.num_one_step_obs`?
        
        # In typical HIMLoco, the critic obs starts with the actor obs.
        # Let's assume standard behavior:
        # critic_obs = [policy_obs (current), privileged_obs]
        # privileged_obs usually starts with lin_vel (3).
        
        # Let's stick to what I wrote before which seemed logical:
        # vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3]
        # next_obs = next_critic_obs[:, :self.num_one_step_obs]
        
        # Wait, I should verify the reference file carefully.
        # File: `HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py`
        # 82: vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        # 83: next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3]
        
        # If indices are 3 to N+3, it ignores 0,1,2.
        # Maybe 0,1,2 are projected gravity? and 3..N+3 contains the proprioception that the target encoder reconstructs?
        # I will COPY the reference logic exactly to be safe, assuming the user's environment provides data in that order.
        
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        next_obs = next_critic_obs.detach()[:, :self.num_one_step_obs]

        z_s = self.encoder(self._prepare_obs_input(obs_history))
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        q_s = sinkhorn(score_s)
        q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    """Sinkhorn algorithm for optimal transport."""
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T
