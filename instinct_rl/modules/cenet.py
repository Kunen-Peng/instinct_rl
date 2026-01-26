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
                 decoder_hidden_dim=[64, 128, 48],
                 init_noise_std=1.0,
                 activation=nn.ELU(),
                 **kwargs):
        """
        cenet: CENet = est_class(self.env.num_encoder_obs=frame_stack * num_single_obs, 
                                 self.env.num_single_obs).to(self.device)
        """
        if kwargs:
            print("CENet.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CENet, self).__init__()
        
        # create encoder: num_encoder_obs -> num_est_h + num_est_v + num_est_z
        self.encoder = MLP(input_dim=num_encoder_obs,
                         hidden_dims=encoder_hidden_dim,
                         output_dim=num_est_h + num_est_v + num_est_z,
                         activation=activation)
        print(f"encoder MLP: {self.encoder}")

        # create decoder: num_est_h + num_est_v + num_est_z -> num_decoder_obs==num_single_obs
        self.decoder = MLP(input_dim=num_est_h + num_est_v+ num_est_z, 
                         hidden_dims=decoder_hidden_dim, 
                         output_dim=num_decoder_obs,
                         activation=activation)
        print(f"decoder MLP: {self.decoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_est_h + num_est_v + num_est_z))

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
        return self.distribution.stddev

    def update_distribution(self, observations):
        mean = self.encoder(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def encode(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def decode(self, decoder_obs):
        est_obs = self.decoder(decoder_obs)
        return est_obs

    def encoder_inference(self, observations):
        encoder_mean = self.encoder(observations)
        return encoder_mean
