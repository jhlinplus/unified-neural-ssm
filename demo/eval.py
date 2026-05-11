"""
Notebook-oriented inspection helpers.

This module exists to make demo-side inspection and plotting code easier to
read. It does not define new model primitives: the underlying smoothing,
reconstruction, decoding, and forecasting APIs already live on the model-side
network objects in `src.models.*.network`.
"""

import torch

from src.models.discrete.network import LatentDiscreteModel
from src.models.ode.network import LatentODEModel
from src.models.sde.network import LatentSDEModel

try:
    from src.models.nowcast.network import LatentNowcastModel
except ModuleNotFoundError:
    LatentNowcastModel = None


_SEQ_LATENT_MODELS = tuple(
    model_type for model_type in (LatentDiscreteModel, LatentNowcastModel, LatentSDEModel)
    if model_type is not None
)


def _batched_inputs(
    x: torch.Tensor,
    mask: torch.Tensor,
    time_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape [L, dim_x], got {tuple(x.shape)}.")
    if mask.shape != x.shape:
        raise ValueError(f"Expected mask.shape == x.shape, got {tuple(mask.shape)} vs {tuple(x.shape)}.")
    if time_steps.ndim != 1 or time_steps.shape[0] != x.shape[0]:
        raise ValueError(
            f"Expected time_steps to have shape [{x.shape[0]}], got {tuple(time_steps.shape)}."
        )
    return x.unsqueeze(0), mask.unsqueeze(0), time_steps.unsqueeze(0)


def _supports_plus_one(network) -> bool:
    return isinstance(network, _SEQ_LATENT_MODELS)


@torch.no_grad()
def infer_context_latent_from_trajectory(
    network,
    x: torch.Tensor,
    mask: torch.Tensor,
    time_steps: torch.Tensor,
) -> torch.Tensor:
    x_b, mask_b, t_b = _batched_inputs(x=x, mask=mask, time_steps=time_steps)

    if isinstance(network, LatentODEModel):
        z_ctx, _, _ = network.smooth(x=x_b, mask=mask_b, time_steps=t_b)
        return z_ctx[0]
    if isinstance(network, _SEQ_LATENT_MODELS):
        mu_zq, _ = network.smooth(x=x_b, mask=mask_b, time_steps=t_b)
        return mu_zq[0]
    raise TypeError(f"Unsupported network type: {type(network).__name__}")


@torch.no_grad()
def reconstruct_context_same_grid_from_trajectory(
    network,
    x: torch.Tensor,
    mask: torch.Tensor,
    time_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_b, mask_b, t_b = _batched_inputs(x=x, mask=mask, time_steps=time_steps)

    if isinstance(network, LatentODEModel):
        x_fit, _, *_ = network.reconstruct(
            x=x_b,
            mask=mask_b,
            time_steps=t_b,
            time_steps_to_reconstruct=t_b,
            mc_samples=0,
        )
        return x_fit[0], time_steps
    if isinstance(network, _SEQ_LATENT_MODELS):
        mu_zq, _ = network.smooth(x=x_b, mask=mask_b, time_steps=t_b)
        x_fit, _ = network.obs_decoder(mu_zq)
        return x_fit[0], time_steps
    raise TypeError(f"Unsupported network type: {type(network).__name__}")


@torch.no_grad()
def reconstruct_context_plus_one_from_trajectory(
    network,
    x: torch.Tensor,
    mask: torch.Tensor,
    time_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(network, LatentODEModel):
        raise Exception(
            "plus-one context reconstruction is not applicable to ODE-based models"
        )
    if not _supports_plus_one(network):
        raise TypeError(f"Unsupported network type: {type(network).__name__}")

    x_b, mask_b, t_b = _batched_inputs(x=x, mask=mask, time_steps=time_steps)
    x_fit, _, *_ = network.reconstruct(
        x=x_b,
        mask=mask_b,
        time_steps=t_b,
        mc_samples=0,
    )
    return x_fit[0], time_steps[1:]


@torch.no_grad()
def forecast_from_trajectory(
    network,
    x: torch.Tensor,
    mask: torch.Tensor,
    time_steps: torch.Tensor,
    time_steps_to_predict: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(network, LatentODEModel):
        x_pred, z_pred, t_pred = network.forecast_from_context(
            x=x,
            mask=mask,
            time_steps=time_steps,
            time_steps_to_predict=time_steps_to_predict,
            anchor_mode="last",
            **kwargs,
        )
        return x_pred, z_pred, t_pred
    if isinstance(network, _SEQ_LATENT_MODELS):
        x_pred, z_pred = network.forecast_from_context(
            x=x,
            mask=mask,
            time_steps=time_steps,
            time_steps_to_predict=time_steps_to_predict,
            **kwargs,
        )
        return x_pred, z_pred, time_steps_to_predict
    raise TypeError(f"Unsupported network type: {type(network).__name__}")
