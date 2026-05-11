from __future__ import annotations

from dataclasses import dataclass

from ..common.config import TrainerConfig

@dataclass(frozen=True)
class SDEModelConfig:
    # Shared across encoder / latent transition / observation decoder
    dim_obs: int
    dim_latent: int

    # Encoder
    enc_rnn_hidden_dim: int
    enc_rnn_num_layers: int
    enc_proj_dim: int

    # Latent transition
    drift_net_hidden_dim: int | list[int]
    diffusion_net_hidden_dim: int | list[int]

    # Observation decoder
    dec_hidden_dim: int | list[int]
    x_sigma2: float
    # Shared regularization across encoder / latent transition / observation decoder
    dropout: float = 0.0
    # Shared latent covariance parameterization for posterior encoder and transition diffusion.
    z_sigma_type: str = "diag"
    enc_direction: str = "backward"

    def __post_init__(self):
        if self.z_sigma_type not in ("const", "diag", "full"):
            raise ValueError(
                f"Unsupported z_sigma_type={self.z_sigma_type}. "
                "Expected one of {'const', 'diag', 'full'}."
            )
        if self.enc_direction not in {"forward", "backward"}:
            raise ValueError(
                f"Unsupported enc_direction={self.enc_direction}. "
                "Expected one of {'forward', 'backward'}."
            )


@dataclass(frozen=True)
class SDEOptConfig(TrainerConfig):
    loss_type: str = "NLL"
    loss_reduction: str = "mean"
    train_strategy: str = "reconstruct"
    kl_mode: str = "mc"
    mc_samples: int = 0
    beta_max: float = 1.0
    warmup_epochs: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.loss_type not in {"MSE", "NLL"}:
            raise ValueError(
                f"Unsupported loss_type={self.loss_type}. "
                "Expected one of {'MSE', 'NLL'}."
            )
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError(
                f"Unsupported loss_reduction={self.loss_reduction}. "
                "Expected one of {'mean', 'sum'}."
            )
        if self.train_strategy != "reconstruct":
            raise ValueError(
                f"SDEOptConfig only supports train_strategy='reconstruct', got {self.train_strategy!r}."
            )
        if self.kl_mode not in {"mc", "exact", "vae"}:
            raise ValueError(
                f"Unsupported kl_mode={self.kl_mode}. "
                "Expected one of {'mc', 'exact', 'vae'}."
            )
        if self.mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {self.mc_samples}.")
