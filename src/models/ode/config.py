from __future__ import annotations

from dataclasses import dataclass

from ..common.config import TrainerConfig

@dataclass(frozen=True)
class ODEModelConfig:
    # Shared across encoder / latent transition / observation decoder
    dim_obs: int
    dim_latent: int

    # Encoder
    enc_rnn_hidden_dim: int
    enc_rnn_num_layers: int
    enc_proj_dim: int

    # Latent transition
    ode_net_hidden_dim: int | list[int]

    # Observation decoder
    dec_hidden_dim: int | list[int]
    x_sigma2: float
    # Shared regularization across encoder / latent transition / observation decoder
    dropout: float = 0.0
    enc_direction: str = "backward"

    def __post_init__(self):
        if self.enc_direction not in {"forward", "backward"}:
            raise ValueError(
                f"Unsupported enc_direction={self.enc_direction}. "
                "Expected one of {'forward', 'backward'}."
            )


@dataclass(frozen=True)
class ODEOptConfig(TrainerConfig):
    loss_mode: str = "iwae"
    loss_type: str = "NLL"
    loss_reduction: str = "mean"
    train_strategy: str = "predict"
    mc_samples: int = 1
    beta_max: float = 1.0
    warmup_epochs: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.loss_mode not in {"elbo", "iwae"}:
            raise ValueError(
                f"Unsupported loss_mode={self.loss_mode}. "
                "Expected one of {'elbo', 'iwae'}."
            )
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
        if self.train_strategy not in {"predict", "reconstruct"}:
            raise ValueError(
                f"Unsupported train_strategy={self.train_strategy}. "
                "Expected one of {'predict', 'reconstruct'}."
            )
        if self.mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {self.mc_samples}.")
