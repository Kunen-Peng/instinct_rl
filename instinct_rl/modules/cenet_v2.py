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
    """Context-Encoder Network (V2).

    Supports two encoder backends:

    * ``encoder_type="rnn"`` (default) — Pre-embedding → GRU → decoupled heads.
      The runner manages hidden states externally.
    * ``encoder_type="mlp"`` — A simple MLP encoder (no recurrence).  This is
      useful for non-recurrent DreamWaQ variants and maintains the same output
      interface (``encode`` / ``encoder_inference`` / ``encoder_inference_recurrent``).
    """

    def __init__(
        self,
        num_encoder_obs,
        num_decoder_obs,
        num_est_v=3,
        num_est_z=16,
        decoder_hidden_dim=[256, 128],
        activation=nn.ELU(),
        # --- Encoder-type selection ---
        encoder_type="rnn",  # "rnn" or "mlp"
        # --- RNN-specific params (ignored when encoder_type="mlp") ---
        rnn_hidden_size=256,
        rnn_num_layers=2,
        pre_embedding_dim=128,
        physical_heads_num_layers=1,
        # --- MLP-specific params (ignored when encoder_type="rnn") ---
        encoder_hidden_dim=None,  # e.g. [256, 128]
        **kwargs,
    ):
        super(CENetV2, self).__init__()

        self.dim_v = num_est_v  # 3
        self.dim_z = num_est_z  # 16
        self.latent_dim = num_est_v + num_est_z           # 19
        self.encoder_output_dim = num_est_v + num_est_z * 2  # 35

        self.raw_encoder_input_dim = num_encoder_obs
        self.encoder_type = encoder_type

        # ------------------------------------------------------------------
        # Build encoder
        # ------------------------------------------------------------------
        if encoder_type == "mlp":
            # ---- MLP encoder ----
            if encoder_hidden_dim is None:
                encoder_hidden_dim = [256, 128]
            self.encoder = MLP(
                input_dim=num_encoder_obs,
                hidden_dims=encoder_hidden_dim,
                output_dim=self.encoder_output_dim,
                activation=activation,
            )
            # RNN-specific attributes are None so callers can check.
            self.pre_embedding = None
            self.rnn = None
            self.v_head = None
            self.z_mean_head = None
            self.z_logstd_head = None

        elif encoder_type == "rnn":
            # ---- RNN (GRU) encoder ----
            self.encoder = None  # not used in RNN mode

            self.pre_embedding = nn.Sequential(
                nn.Linear(num_encoder_obs, pre_embedding_dim),
                nn.LayerNorm(pre_embedding_dim),
                nn.ELU(),
            )
            self.rnn = nn.GRU(
                input_size=pre_embedding_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
            )

            # Decoupled output heads
            if physical_heads_num_layers == 1:
                self.v_head = nn.Linear(rnn_hidden_size, num_est_v)
            else:
                mid = max(32, rnn_hidden_size // 4)
                self.v_head = nn.Sequential(
                    nn.Linear(rnn_hidden_size, mid),
                    nn.ELU(),
                    nn.Linear(mid, num_est_v),
                )
            self.z_mean_head = nn.Linear(rnn_hidden_size, num_est_z)
            self.z_logstd_head = nn.Linear(rnn_hidden_size, num_est_z)

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Must be 'rnn' or 'mlp'.")

        # ------------------------------------------------------------------
        # Decoder (shared by both modes)
        # ------------------------------------------------------------------
        self.decoder = MLP(
            input_dim=self.latent_dim,
            hidden_dims=decoder_hidden_dim,
            output_dim=num_decoder_obs,
            activation=activation,
        )

        # Cached encoder outputs (set during encode())
        self.full_mean = None
        self.z_logvar = None

        Normal.set_default_validate_args = False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
        return self.full_mean  # [Batch, 19]

    @property
    def encoder_logvar(self):
        return self.z_logvar  # [Batch, 16]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heads_from_features(self, features):
        """Apply decoupled heads to RNN output features.

        Args:
            features: ``(Batch, rnn_hidden_size)`` or ``(Batch, SeqLen, rnn_hidden_size)``

        Returns:
            logits: ``(Batch, 35)`` or ``(Batch, SeqLen, 35)``
        """
        v_mean = self.v_head(features)
        z_mean = self.z_mean_head(features)
        z_logstd = self.z_logstd_head(features)
        return torch.cat([v_mean, z_mean, z_logstd], dim=-1)

    def _split_logits_and_sample(self, logits):
        """Split 35-dim encoder logits → sample latent, cache mean/logvar.

        Works for both 2-D ``(B, 35)`` and 3-D ``(B, S, 35)`` inputs.

        Returns:
            full_sampled: ``(..., 19)`` concatenation of ``[v_mean, z_sampled]``
        """
        v_mean = logits[..., : self.dim_v]
        z_params = logits[..., self.dim_v :]
        z_mean, z_log_std = torch.chunk(z_params, 2, dim=-1)

        self.full_mean = torch.cat([v_mean, z_mean], dim=-1)

        z_log_std = torch.clamp(z_log_std, min=-5.0, max=2.0)
        z_std = torch.exp(z_log_std)
        z_sampled = Normal(z_mean, z_std).rsample()

        self.z_logvar = 2 * z_log_std

        return torch.cat([v_mean, z_sampled], dim=-1)

    def _split_logits_mean_only(self, logits):
        """Return deterministic mean ``[v_mean, z_mean]`` from encoder logits."""
        v_mean = logits[..., : self.dim_v]
        z_mean = logits[..., self.dim_v : self.dim_v + self.dim_z]
        return torch.cat([v_mean, z_mean], dim=-1)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, observations, hidden_states=None, masks=None, **kwargs):
        """Encode observations to latent.

        Args:
            observations: ``(B, D)`` or ``(B, SeqLen, D)``
            hidden_states: RNN hidden states (ignored in MLP mode)
            masks: sequence masks (ignored in MLP mode)

        Returns:
            full_sampled: ``(..., 19)``
        """
        if self.encoder_type == "mlp":
            logits = self.encoder(observations)
            return self._split_logits_and_sample(logits)

        # --- RNN path ---
        if observations.dim() == 2:
            rnn_input = observations.unsqueeze(1)
        else:
            rnn_input = observations

        emb = self.pre_embedding(rnn_input)
        if hidden_states is not None:
            rnn_out, _ = self.rnn(emb, hidden_states)
        else:
            rnn_out, _ = self.rnn(emb)

        last_features = rnn_out[:, -1, :]
        logits = self._heads_from_features(last_features)
        return self._split_logits_and_sample(logits)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, decoder_obs):
        return self.decoder(decoder_obs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encoder_inference(self, observations):
        """Inference mode: return deterministic mean ``(v_mean + z_mean)``."""
        if self.encoder_type == "mlp":
            logits = self.encoder(observations)
            return self._split_logits_mean_only(logits)

        # --- RNN path ---
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)

        emb = self.pre_embedding(observations)
        rnn_out, _ = self.rnn(emb, None)
        last_features = rnn_out[:, -1, :]
        logits = self._heads_from_features(last_features)
        return self._split_logits_mean_only(logits)

    def encoder_inference_recurrent(self, observations, hidden_states=None):
        """Inference with explicit hidden-state management.

        In MLP mode ``next_hidden_states`` is always ``None``.
        """
        if self.encoder_type == "mlp":
            logits = self.encoder(observations)
            return self._split_logits_mean_only(logits), None

        # --- RNN path ---
        if observations.dim() == 2:
            rnn_input = observations.unsqueeze(1)
        else:
            rnn_input = observations

        emb = self.pre_embedding(rnn_input)
        rnn_out, next_hidden_states = self.rnn(emb, hidden_states)
        last_features = rnn_out[:, -1, :]
        logits = self._heads_from_features(last_features)
        return self._split_logits_mean_only(logits), next_hidden_states
