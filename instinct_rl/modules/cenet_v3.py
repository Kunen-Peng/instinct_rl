import torch

from instinct_rl.modules.cenet_v2 import CENetV2


class CENetV3(CENetV2):
    """Sequence-aware DreamWaQ encoder.

    V3 keeps the V2 rollout/inference behavior, but also accepts padded
    time-major trajectory batches ``[T, B, D]`` during training.

    In MLP mode (``encoder_type="mlp"``), trajectory handling is unnecessary
    and all calls fall through to the base V2 implementation.
    """

    def _encode_batch_first(self, observations, hidden_states=None):
        """Process a (Batch, SeqLen, Dim) tensor through the RNN encoder."""
        emb = self.pre_embedding(observations)
        if hidden_states is not None:
            rnn_out, _ = self.rnn(emb, hidden_states)
        else:
            rnn_out, _ = self.rnn(emb)

        logits = self._heads_from_features(rnn_out)
        return self._split_logits_and_sample(logits)

    def encode(self, observations, hidden_states=None, masks=None, **kwargs):
        # MLP mode: no trajectory batching needed — delegate to base.
        if self.encoder_type == "mlp":
            return super().encode(observations, **kwargs)

        # RNN mode with padded trajectory batches [T, B, D]
        if observations.dim() == 3 and masks is not None and observations.shape[:2] == masks.shape:
            batch_first_obs = observations.transpose(0, 1).contiguous()
            sampled = self._encode_batch_first(batch_first_obs, hidden_states=hidden_states)
            self.full_mean = self.full_mean.transpose(0, 1).contiguous()
            self.z_logvar = self.z_logvar.transpose(0, 1).contiguous()
            return sampled.transpose(0, 1).contiguous()

        return super().encode(observations, hidden_states=hidden_states, masks=masks, **kwargs)
