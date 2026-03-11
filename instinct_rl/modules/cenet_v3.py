import torch

from instinct_rl.modules.cenet_v2 import CENetV2


class CENetV3(CENetV2):
    """Sequence-aware DreamWaQ encoder.

    V3 keeps the V2 rollout/inference behavior, but also accepts padded
    time-major trajectory batches ``[T, B, D]`` during training.
    """

    def _encode_batch_first(self, observations, hidden_states=None):
        emb = self.pre_embedding(observations)
        if hidden_states is not None:
            rnn_out, _ = self.rnn(emb, hidden_states)
        else:
            rnn_out, _ = self.rnn(emb)

        v_mean = self.v_head(rnn_out)
        z_mean = self.z_mean_head(rnn_out)
        z_log_std = self.z_logstd_head(rnn_out)
        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)

        self.full_mean = torch.cat([v_mean, z_mean], dim=-1)
        self.z_logvar = 2 * z_log_std

        z_std = torch.exp(z_log_std)
        z_sampled = torch.distributions.Normal(z_mean, z_std).rsample()
        return torch.cat([v_mean, z_sampled], dim=-1)

    def encode(self, observations, hidden_states=None, masks=None, **kwargs):
        if observations.dim() == 3 and masks is not None and observations.shape[:2] == masks.shape:
            batch_first_obs = observations.transpose(0, 1).contiguous()
            sampled = self._encode_batch_first(batch_first_obs, hidden_states=hidden_states)
            self.full_mean = self.full_mean.transpose(0, 1).contiguous()
            self.z_logvar = self.z_logvar.transpose(0, 1).contiguous()
            return sampled.transpose(0, 1).contiguous()

        return super().encode(observations, hidden_states=hidden_states, masks=masks, **kwargs)
