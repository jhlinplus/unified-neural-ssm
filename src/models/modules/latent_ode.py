from typing import Callable

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .common import MLP, RNNEncoder, Sampler, GaussianObsDecoder

ODEINT_ATOL = 1e-5
ODEINT_RTOL = 1e-4

class ODEFunc(nn.Module):
    """
    Neural vector field f_theta for latent ODE:
      dz/dt = f_theta(t, z).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
    ) -> None:
        
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        self.ode_gradient_net = MLP(input_dim=input_dim,
                                    output_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    dropout=0.0,
                                    activation="Tanh",
                                    batch_norm=False)

    def forward(self, t: torch.Tensor, y: torch.Tensor, backwards: bool=False) -> torch.Tensor:
        """
        Evaluate latent ODE vector field.

        Args:
          t: [] scalar tensor (unused by this autonomous field, kept for solver API).
          y: latent state tensor with shape `[..., dim_z]`.
          backwards: if True, negate the gradient (currently unused in standard training).

        Returns:
          dy/dt with shape `[..., dim_z]`, i.e., latent time derivative.
        """
        # y.requires_grad = True  # ensure y has gradient
        grad = self.ode_gradient_net(y)
        if backwards:
            grad = -grad
        return grad
    

class ODESolver(nn.Module):
    """
    Thin wrapper over `torchdiffeq.odeint` returning shape [B, L, dim_z] or [S, B, L, dim_z].
    """
    
    def __init__(
        self,
        ode_func: Callable,
        method: str,
        odeint_rtol = ODEINT_RTOL,
        odeint_atol = ODEINT_ATOL
    ) -> None:
        
        super(ODESolver, self).__init__()
        self.ode_method = method
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, initial_point: torch.Tensor, tps: torch.Tensor) -> torch.Tensor:
        """
        Integrate latent states over time grid `tps`.

        Args:
          initial_point: [B, dim_z] or [S, B, dim_z], initial latent states.
          tps: [L], 1D time grid for ODE integration.

        Returns:
          func_vals: [B, L, dim_z] or [S, B, L, dim_z], latent trajectories over `tps`.
        """
        # odeint returns [L, *initial_point.shape]
        func_vals = odeint(self.ode_func, initial_point, tps,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        
        # Move time dimension to middle position
        if initial_point.ndim == 2:  # [B, dim_z]
            func_vals = func_vals.permute(1, 0, 2)  # [L, B, dim_z] → [B, L, dim_z]
        elif initial_point.ndim == 3:  # [S, B, dim_z]
            func_vals = func_vals.permute(1, 2, 0, 3)  # [L, S, B, dim_z] → [S, B, L, dim_z]
        else:
            raise ValueError(f"Expected initial_point [B, dim_z] or [S, B, dim_z], got {initial_point.shape}")
        
        return func_vals


class LatentODEModel(nn.Module):
    """
    Latent Neural ODE model:
      x -> q(z0|x) -> ODE solve z(t) -> decoder x_hat(t).
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        enc_rnn_hidden_dim: int,
        enc_rnn_num_layers: int,
        enc_proj_dim: int,
        ode_net_hidden_dim: int | list[int],
        dec_hidden_dim: int | list[int],
        x_sigma2: float = 0.01,
        enc_direction: str = "backward",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.posterior_encoder = RNNEncoder(
            dim_x=dim_x,
            dim_z=dim_z,
            rnn_hidden_dim=enc_rnn_hidden_dim,
            rnn_num_layers=enc_rnn_num_layers,
            proj_dim=enc_proj_dim,
            sigma_type="diag",
            dropout=dropout,
        )
        self.ode_func = ODEFunc(input_dim=dim_z,
                                hidden_dim=ode_net_hidden_dim)
        self.odesolver = ODESolver(ode_func=self.ode_func, method='dopri5')

        self.obs_decoder = GaussianObsDecoder(
            dim_z=dim_z,
            dim_x=dim_x,
            hidden_dim=dec_hidden_dim,
            x_sigma2=x_sigma2,
            dropout=dropout,
            batch_norm=False,
        )
        self.enc_direction = enc_direction
    
    # ================================================================================
    # Training Methods
    # ================================================================================
    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_reconstruct: torch.Tensor | None = None,
        mc_samples: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct observations on `time_steps_to_reconstruct`.

        Args:
          x: [B, L, dim_x], context observations.
          mask: [B, L, dim_x], observation mask.
          time_steps: [B, L], context timestamps.
          time_steps_to_reconstruct: [B, L_rec] or None (defaults to time_steps).
          mc_samples: number of posterior z0 samples used for reconstruction.
            - 0: use posterior mean (single deterministic path)
            - >=1: sample z0 paths without MC averaging

        Returns:
          x_mean:
            - [B, L_rec, dim_x] when `mc_samples == 0`
            - [S, B, L_rec, dim_x] when `mc_samples >= 1`
          x_sigma2:
            - [B, L_rec, dim_x] when `mc_samples == 0`
            - [S, B, L_rec, dim_x] when `mc_samples >= 1`
          z0_mean: [B, dim_z], posterior mean of initial latent state.
          z0_std: [B, dim_z], posterior std of initial latent state.
          ode_sol:
            - [B, L_rec, dim_z] when `mc_samples == 0`
            - [S, B, L_rec, dim_z] when `mc_samples >= 1`
        """
        if time_steps.ndim != 2:
            raise ValueError(f"Expected time_steps to have 2 dimensions [B, L], got {time_steps.shape}")
        if time_steps_to_reconstruct is None:
            time_steps_to_reconstruct = time_steps
        elif time_steps_to_reconstruct.ndim != 2:
            raise ValueError(
                "Expected time_steps_to_reconstruct to have 2 dimensions [B, L_rec], "
                f"got {time_steps_to_reconstruct.shape}"
            )
        time_steps_to_reconstruct = self._coerce_tps_1d(time_steps_to_reconstruct)
        
        # Encode posterior
        q_mu, q_sigma = self.posterior_encoder(
            x, mask, time_steps, encoding_direction=self.enc_direction
        )  # [B, L, dim_z], [B, L, dim_z]
        z0_mean, z0_std = q_mu[:, 0, :], q_sigma[:, 0, :]
        
        # Stage 1: posterior sampling for z0.
        if mc_samples == 0:
            z0 = z0_mean  # [B, dim_z]
        else:
            z0 = self.sample_posterior_latent_state(
                z0_mean=z0_mean,
                z0_std=z0_std,
                mc_samples=mc_samples,
                take_mc_average=False,
            )  # [S, B, dim_z]
        
        # Stage 2: solve latent ODE on reconstruction grid.
        # [B, L_rec, dim_z] when mc_samples == 0; [S, B, L_rec, dim_z] when mc_samples >= 1.
        ode_sol = self.odesolver(z0, time_steps_to_reconstruct)
        
        # Stage 3: decode latent trajectories.
        # [B, L_rec, dim_x] when mc_samples == 0; [S, B, L_rec, dim_x] when mc_samples >= 1.
        x_mean, x_sigma2 = self.obs_decoder(ode_sol)
        
        return x_mean, x_sigma2, z0_mean, z0_std, ode_sol

    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_reconstruct: torch.Tensor | None = None,
        mc_samples: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Alias of `reconstruct`.

        Args:
          x: [B, L, dim_x], context observations.
          mask: [B, L, dim_x], observation mask.
          time_steps: [B, L], context timestamps.
          time_steps_to_reconstruct: [B, L_rec] or None.
          mc_samples: number of MC samples (0 = use posterior mean, ≥1 = sample and average).
        """
        return self.reconstruct(
            x=x,
            mask=mask,
            time_steps=time_steps,
            time_steps_to_reconstruct=time_steps_to_reconstruct,
            mc_samples=mc_samples,
        )

    # ================================================================================
    # Inference Methods
    # ================================================================================

    @torch.no_grad()
    def smooth(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute smoothed posterior estimates q(z0 | x_{1:L}) and latent path z(t).
        
        For ODE models, this returns the inferred initial state z0 and the 
        deterministic latent trajectory obtained by integrating the ODE from z0.
        
        Args:
          x: [B, L, dim_x], observed trajectory.
          mask: [B, L, dim_x], observation mask.
          time_steps: [B, L], timestamps aligned with `x`.
        
        Returns:
          z_path: [B, L, dim_z], smoothed latent trajectory from ODE integration.
          z0_mean: [B, dim_z], posterior mean of initial latent state.
          z0_std: [B, dim_z], posterior std of initial latent state.
        """
        if time_steps.ndim != 2:
            raise ValueError(f"Expected time_steps to have 2 dimensions [B, L], got {time_steps.shape}")
        time_steps_norm = self._coerce_tps_1d(time_steps)
        q_mu, q_sigma = self.posterior_encoder(
            x, mask, time_steps, encoding_direction=self.enc_direction
        )  # [B, L, dim_z], [B, L, dim_z]
        z0_mean, z0_std = q_mu[:, 0, :], q_sigma[:, 0, :]
        
        # Compute latent path by solving ODE from posterior mean z0
        z_path = self.odesolver(z0_mean, time_steps_norm)  # [B, L, dim_z]
        return z_path, z0_mean, z0_std

    @torch.no_grad()
    def forecast_from_state_batched(
        self,
        initial_state: torch.Tensor,
        tps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper forecast from batched latent anchor states.

        Args:
          initial_state: [B, dim_z], latent anchor states (B can be batch size or number of sampled trajectories).
          tps: [B, L_pred+1], absolute prediction timeline (includes t_init as first element).

        Returns:
          x_samples: [B, L_pred, dim_x], decoded observation predictions.
          z_samples: [B, L_pred, dim_z], latent trajectory predictions.
          tps: [L_pred], prediction timeline aligned with `x_samples`/`z_samples`.
        """
        if tps.ndim != 2:
            raise ValueError(f"Expected tps to have 2 dimensions [B, L_pred+1], got {tps.shape}")

        rollout_grid = self._coerce_tps_1d(tps)  # [L_pred+1], validates shared timeline
        # Solve ODE from each initial state
        ode_sol_full = self.odesolver(initial_state, rollout_grid)  # [B, L_pred+1, dim_z]
        # Drop anchor point at t_init, keep prediction window
        ode_sol = ode_sol_full[:, 1:, :]  # [B, L_pred, dim_z]
        tps = rollout_grid[1:]  # [L_pred]

        # Decode latent predictions
        x_mean_samples, _ = self.obs_decoder(ode_sol)  # [B, L_pred, dim_x]
        return x_mean_samples, ode_sol, tps


    @torch.no_grad()
    def forecast_from_state(
        self,
        initial_state: torch.Tensor,
        tps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forecast from unbatched latent anchor (API matches SDE/discrete).

        Args:
          initial_state: [dim_z], latent anchor state.
          tps: [L_pred+1], absolute prediction timeline (includes t_init as first element).
        
        Returns:
          x_path: [L_pred, dim_x]
          z_path: [L_pred, dim_z]
          tps: [L_pred], prediction timeline aligned with `x_path`/`z_path`.
        """
        if tps.ndim != 1:
            raise ValueError(f"Expected tps to have 1 dimension [L_pred+1], got {tps.shape}")

        initial_state = initial_state.unsqueeze(0)  # [1, dim_z]
        tps_batched = tps.unsqueeze(0)  # [1, L_pred+1]
        x_path, z_path, tps = self.forecast_from_state_batched(
            initial_state=initial_state,
            tps=tps_batched,
        )  # [1, L_pred, dim_x], [1, L_pred, dim_z], [L_pred]
        return x_path.squeeze(0), z_path.squeeze(0), tps  # [L_pred, dim_x], [L_pred, dim_z], [L_pred]

    
    @torch.no_grad()
    def forecast_from_context_batched(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_predict: torch.Tensor,
        n_trajs: int = 1,
        anchor_mode: str = "first",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized forecast from batched observed context.

        Args:
          x: [B, L_obs, dim_x], context observations.`
          mask: [B, L_obs, dim_x], observation mask.
          time_steps: [B, L_obs], context timestamps.
          time_steps_to_predict: [B, L_pred], prediction timestamps.
          n_trajs: number of rollout trajectories.
          anchor_mode: one of {"first", "last"}.
            - "first": anchor rollout at inferred z0 (optionally sampled when n_trajs > 1).
            - "last": anchor rollout at smoothed z_last (optionally sampled when n_trajs > 1).

        Returns:
          x_samples:
            - [B, L_pred, dim_x] when n_trajs == 1
            - [S, B, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [B, L_pred, dim_z] when n_trajs == 1
            - [S, B, L_pred, dim_z] when n_trajs > 1
          tps: [L_pred], prediction timeline aligned with returned paths.
        """
        self._validate_anchor_mode(anchor_mode)
        if n_trajs < 1:
            raise ValueError(f"n_trajs must be >= 1, got {n_trajs}.")
        if time_steps.ndim != 2:
            raise ValueError(
                f"Expected time_steps to have 2 dimensions [B, L_obs], got {time_steps.shape}"
            )
        if time_steps_to_predict.ndim != 2:
            raise ValueError(
                f"Expected time_steps_to_predict to have 2 dimensions [B, L_pred], got {time_steps_to_predict.shape}"
            )

        batch_size = x.shape[0]
        context_tps_1d = self._coerce_tps_1d(time_steps)  # validates shared timeline across batch
        pred_tps_1d = self._coerce_tps_1d(time_steps_to_predict)  # validates shared prediction timeline across batch

        z_path_ctx, z0_mean, z0_std = self.smooth(
            x=x,
            mask=mask,
            time_steps=time_steps,
        )  # [B, L_obs, dim_z], [B, dim_z], [B, dim_z]

        if anchor_mode == "first":
            anchor_mean = z0_mean  # [B, dim_z]
            anchor_std = z0_std  # [B, dim_z]
            t_anchor = context_tps_1d[0]  # scalar
        else:
            anchor_mean = z_path_ctx[:, -1, :]  # [B, dim_z]
            anchor_std = None
            if n_trajs > 1:
                # Use encoder's last-step posterior scale for stochastic sampling.
                _, q_sigma = self.posterior_encoder(
                    x, mask, time_steps, encoding_direction=self.enc_direction
                )  # [B, L_obs, dim_z]
                anchor_std = q_sigma[:, -1, :]  # [B, dim_z]
            t_anchor = context_tps_1d[-1]  # scalar

        tps_forecast = torch.cat([t_anchor.unsqueeze(0), pred_tps_1d], dim=0)  # [L_pred+1]
        tps = tps_forecast[1:]  # [L_pred]
        tps_forecast_batched = tps_forecast.unsqueeze(0).expand(batch_size, -1)  # [B, L_pred+1]

        if n_trajs == 1:
            return self.forecast_from_state_batched(
                initial_state=anchor_mean,
                tps=tps_forecast_batched,
            )

        initial_state = self.sample_posterior_latent_state(
            z0_mean=anchor_mean,
            z0_std=anchor_std if anchor_std is not None else z0_std,
            mc_samples=n_trajs,
            take_mc_average=False,
        )  # [S, B, dim_z]

        initial_state_flat = initial_state.reshape(n_trajs * batch_size, self.dim_z)  # [S*B, dim_z]
        tps_forecast_flat = tps_forecast_batched.unsqueeze(0).expand(n_trajs, -1, -1).reshape(
            n_trajs * batch_size, tps_forecast_batched.shape[1]
        )  # [S*B, L_pred+1]
        x_flat, z_flat, _ = self.forecast_from_state_batched(
            initial_state=initial_state_flat,
            tps=tps_forecast_flat,
        )  # [S*B, L_pred, dim_x], [S*B, L_pred, dim_z]
        x_samples = x_flat.view(n_trajs, batch_size, x_flat.shape[1], self.dim_x)
        z_samples = z_flat.view(n_trajs, batch_size, z_flat.shape[1], self.dim_z)
        return x_samples, z_samples, tps
    

    @torch.no_grad()
    def forecast_from_context(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_predict: torch.Tensor,
        n_trajs: int = 1,
        anchor_mode: str = "first",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forecast from unbatched observed context (API matches SDE/discrete).

        Args:
          x: [L_obs, dim_x], context observations.
          mask: [L_obs, dim_x], observation mask.
          time_steps: [L_obs], context timestamps.
          time_steps_to_predict: [L_pred], prediction timestamps (must be > time_steps[-1]).
          n_trajs: number of rollout trajectories (for ODE, samples multiple z0 values).
          anchor_mode: one of {"first", "last"}.
            - "first": anchor rollout at inferred z0 (optionally sampled when n_trajs > 1).
            - "last": anchor rollout at smoothed z_last (optionally sampled when n_trajs > 1).

        Returns:
          x_path: [L_pred, dim_x] or [S, L_pred, dim_x] (n_trajs > 1)
          z_path: [L_pred, dim_z] or [S, L_pred, dim_z] (n_trajs > 1)
          tps: [L_pred], prediction timeline aligned with returned paths.
        """
        if time_steps.ndim != 1:
            raise ValueError(f"Expected time_steps to have 1 dimension [L_obs], got {time_steps.shape}")
        if time_steps_to_predict.ndim != 1:
            raise ValueError(
                f"Expected time_steps_to_predict to have 1 dimension [L_pred], got {time_steps_to_predict.shape}"
            )

        x_samples, z_samples, tps = self.forecast_from_context_batched(
            x=x.unsqueeze(0),
            mask=mask.unsqueeze(0),
            time_steps=time_steps.unsqueeze(0),
            time_steps_to_predict=time_steps_to_predict.unsqueeze(0),
            n_trajs=n_trajs,
            anchor_mode=anchor_mode,
        )
        if n_trajs == 1:
            return x_samples.squeeze(0), z_samples.squeeze(0), tps
        return x_samples[:, 0], z_samples[:, 0], tps

    # ================================================================================
    # Internal Helpers
    # ================================================================================

    def sample_posterior_latent_state(
        self,
        z0_mean: torch.Tensor,
        z0_std: torch.Tensor,
        mc_samples: int = 1,
        take_mc_average: bool = False,
    ) -> torch.Tensor:
        """
        Sample latent initial states from q(z0|x).

        Args:
          z0_mean: [B, dim_z], posterior mean of z0.
          z0_std: [B, dim_z], posterior std of z0.
          mc_samples: number of posterior samples (must be >= 1).
          take_mc_average: whether to average over sample dimension.

        Returns:
          sampled_z0:
            - [S, B, dim_z] when `take_mc_average=False`
            - [B, dim_z] when `take_mc_average=True`
        """
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}.")
        return Sampler.sample_normal(
            z0_mean,
            z0_std ** 2,
            mc_samples=mc_samples,
            take_mc_average=take_mc_average,
        )
    
    @staticmethod
    def _coerce_tps_1d(time_steps: torch.Tensor) -> torch.Tensor:
        """
        Extract or ensure 1D time grid (ODE solver requirement).

        Args:
          time_steps: [L] or [B, L]

        Returns:
          [L], 1D time grid extracted from input.
        """
        if time_steps.ndim == 1:
            return time_steps
        if time_steps.ndim == 2:
            # ODE solver requires one shared time grid for all batch elements.
            # Allow tiny floating-point drift across rows from dataset/index conversions.
            if not torch.allclose(
                time_steps,
                time_steps[0:1].expand_as(time_steps),
                atol=1e-6,
                rtol=1e-5,
            ):
                raise ValueError(
                    "ODE solver requires a shared 1D time grid; received batched "
                    "time_steps with different timelines across samples."
                )
            return time_steps[0]
        raise ValueError(
            f"Expected time_steps to have 1 or 2 dimensions, got {time_steps.shape}"
        )

    @staticmethod
    def _validate_anchor_mode(anchor_mode: str) -> None:
        if anchor_mode not in {"first", "last"}:
            raise ValueError(
                f"Unsupported anchor_mode={anchor_mode}. "
                "Expected one of {'first', 'last'}."
            )
