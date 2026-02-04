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


class CENet(nn.Module):
    def __init__(self,
                 num_encoder_obs,
                 num_decoder_obs,
                 num_est_h=0, 
                 num_est_v=3, 
                 num_est_z=16,
                 encoder_hidden_dim=[512, 256, 128],
                 decoder_hidden_dim=[64, 48],
                 activation=nn.ELU(),
                 **kwargs):
        super(CENet, self).__init__()
        
        # [修正 1] 保存维度参数，供 encode 使用
        self.dim_v = num_est_v  # 3
        self.dim_z = num_est_z  # 16
        
        # 实际的 Latent 维度 (Policy 和 Decoder 看到的维度) = 3 + 16 = 19
        self.latent_dim = num_est_h + num_est_v + num_est_z
        
        # Encoder 输出头维度 = v_mean(3) + z_mean(16) + z_logstd(16) = 35
        self.encoder_output_dim = self.dim_v + self.dim_z * 2
        
        # [修正 2] Encoder 输出 35 维
        self.encoder = MLP(input_dim=num_encoder_obs,
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
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def encoder_mean(self):
        return self.full_mean # [Batch, 19]

    @property
    def encoder_logvar(self):
        return self.z_logvar # [Batch, 16] (只包含 z 的 logvar)

    def encode(self, observations, **kwargs):
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
        
        # 重参数化采样
        # z_sampled: [Batch, 16]
        z_sampled = Normal(z_mean, z_std).rsample() # 建议用 rsample 保留重参数化梯度
        
        # 拼接最终的采样向量 (Decoder 输入) -> [Batch, 19]
        full_sampled = torch.cat([v_mean, z_sampled], dim=-1)
        
        # 保存 z 的 log_std (x2 转为 logvar) 供 KL Loss 使用
        self.z_logvar = 2 * z_log_std 
        
        return full_sampled

    def decode(self, decoder_obs):
        # decoder_obs 应该是 19 维的 full_sampled
        est_obs = self.decoder(decoder_obs)
        return est_obs

    def encoder_inference(self, observations):
        """推理模式：只返回确定性的均值"""
        logits = self.encoder(observations) # [Batch, 35]
        
        # 提取 v_mean
        v_mean = logits[:, :self.latent_dim]
        
        return v_mean