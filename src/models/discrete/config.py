from __future__ import annotations

from dataclasses import dataclass

from ..common.config import TrainerConfig


@dataclass(frozen=True)
class DiscreteModelConfig:
    dim_obs: int
    dim_latent: int

    enc_proj_dim: int
    enc_rnn_hidden_dim: int
    enc_rnn_num_layers: int
    mask_processor_kernel_size: int

    ar_proj_hidden_dim: int
    ar_mean_net_hidden_dim: int | list[int]
    ar_sigma_net_hidden_dim: int | list[int]

    dec_hidden_dim: int | list[int]
    x_sigma2: float

    z_sigma_type: str = "const"
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
    train_strategy: str = "reconstruct"
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
        if self.train_strategy != "reconstruct":
            raise ValueError(
                "DiscreteOptConfig only supports train_strategy='reconstruct', "
                f"got {self.train_strategy!r}."
            )
        if self.mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {self.mc_samples}.")
