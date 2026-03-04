import torch
import torch.nn as nn
from torch.distributions import Normal

# Module-level import to avoid repeated dynamic imports inside forward()
from instinct_rl.modules.him_actor_critic import term_major_to_time_major


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class FeedForward(nn.Module):
    """Standard Feed-Forward block used in MLP-Mixer (matching reference implementation)."""
    def __init__(self, dim, hidden_dim, activation=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """MLP-Mixer block following the canonical architecture from the reference.
    
    Input shape:  (Batch, NumPatches, Dim)
    In our domain: (Batch, NumTimeSteps=H, FeaturesPerStep=D)
    
    - Token-mixing: mixes across time-steps (dim=1), i.e. "which time-step info to attend to"
      Pre-norm(D) → Transpose → FF(H, token_dim) → Transpose back → Residual
    - Channel-mixing: mixes across features (dim=-1), i.e. "which sensor features to combine"
      Pre-norm(D) → FF(D, channel_dim) → Residual
    """
    def __init__(self, num_patches, dim, token_dim=64, channel_dim=128, activation=nn.ELU()):
        super().__init__()
        # Token-mixing: mix across patches/time-steps (dim=1)
        # Pre-norm on feature dim → transpose → FF(num_patches) → transpose back
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = FeedForward(num_patches, token_dim, activation)
        
        # Channel-mixing: mix across features/channels (dim=-1)
        # Pre-norm on feature dim → FF(dim) → residual
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = FeedForward(dim, channel_dim, activation)

    def forward(self, x):
        # x: (Batch, H, D) where H=time-steps, D=features-per-step
        
        # 1. Token Mixing: mix across time-steps
        #    norm over feature dim → transpose to (B, D, H) → FF on H dim → transpose back
        residual = x
        x_norm = self.norm1(x)                          # (B, H, D) — norm over D
        x_t = x_norm.transpose(1, 2).contiguous()       # (B, D, H)
        x_t = self.token_mix(x_t)                       # (B, D, H) — FF mixes across H
        x = residual + x_t.transpose(1, 2).contiguous() # (B, H, D) — residual connection
        
        # 2. Channel Mixing: mix across features
        #    norm over feature dim → FF on D dim → residual
        x = x + self.channel_mix(self.norm2(x))          # (B, H, D)

        return x


class MLPMixerEncoder(nn.Module):
    """MLP-Mixer encoder for proprioceptive history observations.
    
    Reshapes flattened history obs into (Batch, H, D), applies MixerBlock(s),
    then projects to the desired output dimension (CENet latent head).
    """
    def __init__(self, num_history_steps, obs_dim_per_step, output_dim, 
                 token_dim=64, channel_dim=128, activation=nn.ELU(), 
                 obs_layout="time_major", term_dims=None, depth=1):
        super().__init__()
        self.num_history_steps = num_history_steps  # H (num_patches)
        self.obs_dim_per_step = obs_dim_per_step    # D (dim per patch)
        self.obs_layout = obs_layout
        self.term_dims = term_dims
        
        # Stack of MixerBlocks (depth=1 is sufficient for H=5, D=45)
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_patches=num_history_steps,
                dim=obs_dim_per_step,
                token_dim=token_dim,
                channel_dim=channel_dim,
                activation=activation
            )
            for _ in range(depth)
        ])
        
        # Final norm (following reference: norm after all mixer blocks)
        self.layer_norm = nn.LayerNorm(obs_dim_per_step)
        
        # Project to output (v_mean + z_mean + z_logstd = 35 dim)
        self.head = nn.Linear(num_history_steps * obs_dim_per_step, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Strip estimator if appended
        raw_history_dim = self.num_history_steps * self.obs_dim_per_step
        raw_x = x[:, :raw_history_dim]
        
        # Convert IsaacLab term-major layout to time-major if needed
        if self.obs_layout == "term_major" and self.term_dims is not None:
            raw_x = term_major_to_time_major(raw_x, self.term_dims, self.num_history_steps)
        
        # 1. Reshape: (B, H*D) → (B, H, D)
        x = raw_x.reshape(batch_size, self.num_history_steps, self.obs_dim_per_step)
        
        # 2. Apply MixerBlock(s)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        # 3. Final norm + flatten + project
        x = self.layer_norm(x)
        x = x.reshape(batch_size, -1)
        return self.head(x)


from .actor_critic_recurrent import Memory

class CENet(nn.Module):
    def __init__(self,
                 num_encoder_obs,
                 num_decoder_obs,
                 num_est_h=0, 
                 num_est_v=3, 
                 num_est_z=16,
                 encoder_hidden_dim=[256, 128],
                 decoder_hidden_dim=[256, 128],
                 activation=nn.ELU(),
                 rnn_type=None, # [New]
                 rnn_hidden_size=256, # [New]
                 rnn_num_layers=1, # [New]
                 encoder_type="mlp", # [New] MLPMixer encoder support
                 num_history_steps=5, # [New]
                 obs_layout="time_major", # [New] Handling history format
                 term_dims=None, # [New]
                 mixer_hidden_dim=64, # [New] token_dim for MixerBlock (mixing across time steps)
                 mixer_channel_dim=128, # [New] channel_dim for MixerBlock (mixing across features)
                 mixer_depth=1, # [New] Number of MixerBlocks
                 **kwargs):
        super(CENet, self).__init__()
        
        # [修正 1] 保存维度参数，供 encode 使用
        self.dim_v = num_est_v  # 3
        self.dim_z = num_est_z  # 16
        
        # 实际的 Latent 维度 (Policy 和 Decoder 看到的维度) = 3 + 16 = 19
        self.latent_dim = num_est_h + num_est_v + num_est_z
        
        # Encoder 输出头维度 = v_mean(3) + z_mean(16) + z_logstd(16) = 35
        self.encoder_output_dim = self.dim_v + self.dim_z * 2
        
        self.encoder_type = encoder_type

        # [New] RNN Support
        self.rnn = None
        # Mixer replaces RNN. If mixer is active, no RNN.
        if self.encoder_type == "mixer":
            rnn_type = None
            
        if rnn_type is not None:
            self.rnn = Memory(input_size=num_encoder_obs, 
                              type=rnn_type, 
                              num_layers=rnn_num_layers, 
                              hidden_size=rnn_hidden_size)
            # Encoder input dim becomes rnn output dim (hidden size)
            encoder_input_dim = rnn_hidden_size
        else:
            encoder_input_dim = num_encoder_obs

        # [修正 2] Encoder 输出 35 维
        if self.encoder_type == "mixer":
            obs_dim_per_step = num_encoder_obs // num_history_steps
            self.encoder = MLPMixerEncoder(
                num_history_steps=num_history_steps,
                obs_dim_per_step=obs_dim_per_step,
                output_dim=self.encoder_output_dim, # 35
                token_dim=mixer_hidden_dim,      # hidden dim for time-step mixing FF
                channel_dim=mixer_channel_dim,   # hidden dim for feature mixing FF
                activation=activation,
                obs_layout=obs_layout,
                term_dims=term_dims,
                depth=mixer_depth
            )
        else:
            self.encoder = MLP(input_dim=encoder_input_dim,
                               hidden_dims=encoder_hidden_dim,
                               output_dim=self.encoder_output_dim, # 35
                               activation=activation)

        # [修正 3] Decoder 输入 19 维 (self.latent_dim)
        self.decoder = MLP(input_dim=self.latent_dim, # 19
                           hidden_dims=decoder_hidden_dim, 
                           output_dim=num_decoder_obs,
                           activation=activation)

        # 初始化缓存变量
        self.full_mean = None
        self.z_logvar = None
        
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        if self.rnn is not None:
            self.rnn.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def encoder_mean(self):
        return self.full_mean # [Batch, 19]

    @property
    def encoder_logvar(self):
        return self.z_logvar # [Batch, 16] (只包含 z 的 logvar)

    @property
    def raw_encoder_input_dim(self):
        """Returns the raw observation input size (before any augmentation)."""
        if self.encoder_type == "mixer":
            return self.encoder.num_history_steps * self.encoder.obs_dim_per_step
        elif self.rnn is not None:
            return self.rnn.rnn.input_size
        else:
            return self.encoder.model[0].in_features

    def encode(self, observations, hidden_states=None, masks=None, **kwargs):
        # [New] RNN Processing
        next_hidden_states = None
        if self.rnn is not None:
             # Memory module logic:
             # If hidden_states is NOT None -> Batch Mode. Expects Sequence input (L, N, H).
             # If hidden_states IS None -> Inference Mode. Handles unsqueeze internally (1, N, H).
             
             if hidden_states is not None:
                 # Check if observations is 2D (N, H) -> Unsqueeze to (1, N, H)
                 if observations.dim() == 2:
                     rnn_input = observations.unsqueeze(0)
                     rnn_out = self.rnn(rnn_input, masks, hidden_states)
                     # rnn_out will be (1, N, H) (or L, N, H)
                     if rnn_out.dim() == 3:
                         rnn_out = rnn_out.squeeze(0)
                 else:
                     # Already 3D or more? Just pass through
                     rnn_out = self.rnn(observations, masks, hidden_states)
             else:
                 # Inference mode handles unsqueeze internally
                 rnn_out = self.rnn(observations, masks, hidden_states)
             
             logits = self.encoder(rnn_out)
        else:
             logits = self.encoder(observations) # [Batch, 35]
        
        # 切片逻辑
        # 1. v_mean: 前 3 维
        v_mean = logits[:, :self.dim_v]
        
        # 2. z_params: 后 32 维
        z_params = logits[:, self.dim_v:]
        z_mean, z_log_std = torch.chunk(z_params, 2, dim=-1)
        
        # 拼接完整的 Mean (用于 Loss 计算) -> [Batch, 19]
        self.full_mean = torch.cat([v_mean, z_mean], dim=-1)
        
        # 处理 Std 和 采样 (仅针对 z)
        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)
        z_std = torch.exp(z_log_std)
        
        # 重参数化采样 z: [Batch, 16]
        z_sampled = Normal(z_mean, z_std).rsample()
        
        # Concat into final latent (Decoder input) -> [Batch, 19]
        full_sampled = torch.cat([v_mean, z_sampled], dim=-1)
        
        # 保存 z 的 log_std (x2 转为 logvar) 供 KL Loss 使用
        self.z_logvar = 2 * z_log_std
        
        return full_sampled

    def decode(self, decoder_obs):
        # decoder_obs 应该是 19 维的 full_sampled
        est_obs = self.decoder(decoder_obs)
        return est_obs

    def encoder_inference(self, observations):
        """推理模式：只返回确定性的均值 (v_mean + z_mean)"""
        if self.rnn is not None:
            rnn_input = observations
            if rnn_input.dim() == 2:
                rnn_input = rnn_input.unsqueeze(0)
            rnn_out, _ = self.rnn.rnn(rnn_input, None)
            observations = rnn_out.squeeze(0)
        
        logits = self.encoder(observations)  # [Batch, 35]
        
        # Return only the mean part (v_mean + z_mean), drop z_logstd
        v_mean = logits[:, :self.latent_dim]
        return v_mean
    
    # [新增] 显式支持 state 的 inference 接口
    def encoder_inference_recurrent(self, observations, hidden_states=None):
        if self.rnn is not None:
             # Manually handle RNN to get next_hidden_states
             # observations: (N, Dim)
             rnn_input = observations
             if rnn_input.dim() == 2:
                  rnn_input = rnn_input.unsqueeze(0) # (1, N, Dim)
             
             # Call underlying RNN directly: self.rnn is Memory, self.rnn.rnn is GRU/LSTM
             rnn_out, next_hidden_states = self.rnn.rnn(rnn_input, hidden_states)
             
             # Squeeze output back for MLP
             rnn_out = rnn_out.squeeze(0)
             
             # rnn_out: (N, Hidden)
             observations = rnn_out
        else:
             next_hidden_states = None
             
        logits = self.encoder(observations)
        v_mean = logits[:, :self.latent_dim]
        return v_mean, next_hidden_states