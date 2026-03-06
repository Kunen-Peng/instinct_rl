import torch
import torch.nn as nn
from torch.distributions import Normal


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


class CENetV2(nn.Module):
    def __init__(self,
                 num_encoder_obs,
                 num_decoder_obs,
                 num_est_h=0, 
                 num_est_v=3, 
                 num_est_z=16,
                 decoder_hidden_dim=[256, 128],
                 activation=nn.ELU(),
                 rnn_hidden_size=256,  # GRU 隐藏层维度，可通过 cenet 配置覆盖
                 rnn_num_layers=2,     # GRU 层数，可通过 cenet 配置覆盖
                 pre_embedding_dim=128, # Pre-Embedding 输出维度（GRU 的 input_size）
                 physical_heads_num_layers=1, # 速度和高度头的层数（1代表仅Linear, 2代表Linear->ELU->Linear）
                 **kwargs):
        super(CENetV2, self).__init__()
        
        self.dim_v = num_est_v  # 3
        self.dim_z = num_est_z  # 16
        
        # 实际的 Latent 维度 (Policy 和 Decoder 看到的维度) = 3 + 16 = 19
        self.latent_dim = num_est_h + num_est_v + num_est_z
        
        self.raw_encoder_input_dim = num_encoder_obs
        
        # 阶段一：前置嵌入 (Pre-Embedding)
        # 将原始观测映射到固定维度，作为 GRU 的输入序列
        self.pre_embedding = nn.Sequential(
            nn.Linear(num_encoder_obs, pre_embedding_dim),
            nn.LayerNorm(pre_embedding_dim),
            nn.ELU()
        )
        
        # 阶段一：升级 GRU 主干（层数和隐藏维度由配置文件控制）
        self.rnn = nn.GRU(
            input_size=pre_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        
        # 阶段一：彻底拆分解耦输出头 (Decoupled Heads)
        # 所有 Head 的输入维度 = rnn_hidden_size
        # 速度预测头 (只处理物理量，根据设计哲学可能需要更深网络映射抽象特征)
        if physical_heads_num_layers == 1:
            self.v_head = nn.Linear(rnn_hidden_size, 3)
        else:
            self.v_head = nn.Sequential(
                nn.Linear(rnn_hidden_size, max(32, rnn_hidden_size // 4)),
                nn.ELU(),
                nn.Linear(max(32, rnn_hidden_size // 4), 3)
            )
        
        # 隐变量均值头
        self.z_mean_head = nn.Linear(rnn_hidden_size, num_est_z)
        
        # 隐变量方差头
        self.z_logstd_head = nn.Linear(rnn_hidden_size, num_est_z)
        
        # 足端高度预测头 (只处理物理量)
        if physical_heads_num_layers == 1:
            self.h_head = nn.Linear(rnn_hidden_size, 4)
        else:
            self.h_head = nn.Sequential(
                nn.Linear(rnn_hidden_size, max(32, rnn_hidden_size // 4)),
                nn.ELU(),
                nn.Linear(max(32, rnn_hidden_size // 4), 4)
            )

        # Decoder 输入 19 维 (self.latent_dim=19)
        self.decoder = MLP(input_dim=self.latent_dim, 
                           hidden_dims=decoder_hidden_dim, 
                           output_dim=num_decoder_obs,
                           activation=activation)

        # 初始化缓存变量
        self.full_mean = None
        self.z_logvar = None
        self.h_pred = None
        
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        # Memory resetting is typically handled externally in the runner,
        # but we preserve the interface here.
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def encoder_mean(self):
        return self.full_mean # [Batch, 19]

    @property
    def encoder_logvar(self):
        return self.z_logvar # [Batch, 16]
        
    @property
    def encoder_h_pred(self):
        return self.h_pred # [Batch, 4]

    def encode(self, observations, hidden_states=None, masks=None, **kwargs):
        """
        observations: (Batch, Dim) or (Batch, Seq_Len, Dim)
        hidden_states: (Num_layers, Batch, Hidden_size) expected
        """
        # Ensure observations is 3D for batch_first=True RNN
        if observations.dim() == 2:
            rnn_input = observations.unsqueeze(1)  # (Batch, 1, Dim)
        elif observations.dim() == 3 and not self.rnn.batch_first:
             # Just in case, though we hardcoded batch_first=True.
             pass
        else:
            rnn_input = observations

        # 1. Pre-Embedding
        emb = self.pre_embedding(rnn_input) # (Batch, Seq_len, 128)
        
        # 2. GRU
        if hidden_states is not None:
             rnn_out, _ = self.rnn(emb, hidden_states)
        else:
             rnn_out, _ = self.rnn(emb)
             
        # Extract the last feature for each sequence
        last_features = rnn_out[:, -1, :] # (Batch, 256)
        
        # 3. Decoupled Heads
        v_mean = self.v_head(last_features)
        z_mean = self.z_mean_head(last_features)
        z_log_std = self.z_logstd_head(last_features)
        self.h_pred = self.h_head(last_features)  # Save height prediction
        
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
        # Ensure 3D (Batch, 1, Dim)
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
            
        emb = self.pre_embedding(observations)
        rnn_out, _ = self.rnn(emb, None)
        last_features = rnn_out[:, -1, :]
        
        v_mean = self.v_head(last_features)
        z_mean = self.z_mean_head(last_features)
        
        # 阶段四：推理只需拼接 v_mean 和 z_mean
        return torch.cat((v_mean, z_mean), dim=-1)
    
    def encoder_inference_recurrent(self, observations, hidden_states=None):
        """显式支持 state 的 inference 接口"""
        if observations.dim() == 2:
            rnn_input = observations.unsqueeze(1) # (Batch, 1, Dim)
        else:
            rnn_input = observations
            
        emb = self.pre_embedding(rnn_input)
        
        # hidden_states: (Num_layers, Batch, Hidden_size) = (2, B, 256)
        rnn_out, next_hidden_states = self.rnn(emb, hidden_states)
        
        last_features = rnn_out[:, -1, :] # (Batch, 256)
        
        v_mean = self.v_head(last_features)
        z_mean = self.z_mean_head(last_features)
        
        # 阶段四：推理只需拼接 v_mean 和 z_mean
        v_full_mean = torch.cat((v_mean, z_mean), dim=-1)
        
        return v_full_mean, next_hidden_states
