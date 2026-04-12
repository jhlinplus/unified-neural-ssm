from __future__ import annotations

from dataclasses import dataclass

from .trainer import TrainerConfig

@dataclass(frozen=True)
class DiscreteModelConfig:
    # Shared across encoder / latent transition / observation decoder
    dim_obs: int
    dim_latent: int

    # Encoder
    enc_proj_dim: int
    enc_rnn_hidden_dim: int
    enc_rnn_num_layers: int
    mask_processor_kernel_size: int

    # Latent transition
    ar_proj_hidden_dim: int
    ar_mean_net_hidden_dim: int | list[int]
    ar_sigma_net_hidden_dim: int | list[int]

    # Observation decoder
    dec_hidden_dim: int | list[int]
    x_sigma2: float

    # Shared latent covariance parameterization for posterior encoder and transition diffusion.
    z_sigma_type: str = "const"
    
    # Shared regularization across encoder / latent transition / observation decoder
    dropout: float = 0.0

    def __post_init__(self):
        if self.z_sigma_type not in {"const", "diag"}:
            raise ValueError(
                f"Unsupported z_sigma_type={self.z_sigma_type}. "
                "Expected one of {'const', 'diag'}."
            )


@dataclass(frozen=True)
class DiscreteOptConfig(TrainerConfig):
    loss_type: str = "NLL"
    loss_reduction: str = "mean"
    mc_samples: int = 1
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
        if self.mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {self.mc_samples}.")

