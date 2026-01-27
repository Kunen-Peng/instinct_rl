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
            
            self.latent_dim = num_est_h + num_est_v + num_est_z
            
            # [修改 1] Encoder 输出维度翻倍： mean + log_std
            self.encoder = MLP(input_dim=num_encoder_obs,
                            hidden_dims=encoder_hidden_dim,
                            output_dim=self.latent_dim * 2, # <--- 翻倍
                            activation=activation)

            self.decoder = MLP(input_dim=self.latent_dim, 
                            hidden_dims=decoder_hidden_dim, 
                            output_dim=num_decoder_obs,
                            activation=activation)

            # 删除 self.std，不再需要全局固定方差
            
            self.distribution = None
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
        return self.distribution.mean

    @property
    def encoder_logvar(self):
        # [修改 2] 正确返回 log_var (即 2 * log_std)
        # stddev = exp(log_std) -> log(variance) = 2 * log_std
        return 2 * torch.log(self.distribution.stddev)

    def update_distribution(self, observations):
            # [修改 3] 从网络输出中切分出 mean 和 log_std
            logits = self.encoder(observations)
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            
            # [可选优化] 限制 log_std 范围，防止梯度爆炸 (论文提到的 Sigma Clamping 可以通过限制 log_std 实现)
            # 对应 sigma 范围 [exp(-5), exp(2)] approx [0.006, 7.3]
            log_std = torch.clamp(log_std, min=-5.0, max=2.0) 
            std = torch.exp(log_std)
            
            self.distribution = Normal(mean, std)

    def encode(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def decode(self, decoder_obs):
        est_obs = self.decoder(decoder_obs)
        return est_obs

    def encoder_inference(self, observations):
            # 1. 获取包含 mean 和 log_std 的原始输出
        logits = self.encoder(observations)
        
        # 2. 切分出均值 (mean) 和 对数标准差 (log_std)
        # 我们只需要 mean 用于推理 (Inference)
        mean, _ = torch.chunk(logits, 2, dim=-1)
        
        return mean
