from __future__ import annotations

from dataclasses import dataclass

from ..common.config import TrainerConfig


@dataclass(frozen=True)
class NowcastModelConfig:
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
class NowcastOptConfig(TrainerConfig):
    loss_type: str = "NLL"
    loss_reduction: str = "mean"
    train_strategy: str = "reconstruct"
    mc_samples: int = 1
    beta_max: float = 1.0
    warmup_epochs: int | None = None
    pseudo_guidance_mask_prob: float | tuple[float, float] | None = None
    unguided_sample_ratio: float = 0.0
    allow_all_zero_target_mask: bool = False

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
                "NowcastOptConfig only supports train_strategy='reconstruct', "
                f"got {self.train_strategy!r}."
            )
        if self.mc_samples < 0:
            raise ValueError(f"mc_samples must be >= 0, got {self.mc_samples}.")

        if not 0.0 <= self.unguided_sample_ratio <= 1.0:
            raise ValueError(
                "unguided_sample_ratio must satisfy 0 <= unguided_sample_ratio <= 1."
            )

        prob = self.pseudo_guidance_mask_prob
        if prob is None:
            return
        if isinstance(prob, float):
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"pseudo_guidance_mask_prob={prob} must be in [0, 1] when given as a float."
                )
            return
        if isinstance(prob, tuple):
            if len(prob) != 2:
                raise ValueError(
                    "pseudo_guidance_mask_prob must be a 2-tuple when given as a range."
                )
            low, high = prob
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(
                    "pseudo_guidance_mask_prob tuple must satisfy 0 <= low <= high <= 1."
                )
            return
        raise TypeError(
            "pseudo_guidance_mask_prob must be one of float, tuple[float, float], or None."
        )
