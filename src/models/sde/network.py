import torch
import torch.nn as nn

from ..common.nn import GaussianObsDecoder, MLP, RNNEncoder, STD_MIN_SCALE, Sampler


def _triangle_vec_to_lower(vec: torch.Tensor, dim_z: int) -> torch.Tensor:
    """
    Convert packed lower-triangle vectors into dense lower-triangular matrices.

    Args:
      vec: [..., tri_dim], where tri_dim = dim_z * (dim_z + 1) // 2.
      dim_z: latent dimension Z.

    Returns:
      lower: [..., dim_z, dim_z]
    """
    tri_inds = torch.tril_indices(dim_z, dim_z, device=vec.device)
    lower = torch.zeros(*vec.shape[:-1], dim_z, dim_z, device=vec.device, dtype=vec.dtype)
    lower[..., tri_inds[0], tri_inds[1]] = vec
    return lower


class DriftNet(nn.Module):
    """
    Drift parameterization for latent SDE:
      mu_hat_t = f_theta(t, z_t).

    Shapes:
      input:  t scalar-like, z [N, dim_z]
      output: mu_hat [N, dim_z]
    """

    def __init__(self, dim_z: int, hidden_dim: int | list[int], dropout: float = 0.0):
        super().__init__()
        self.net = MLP(
            input_dim=dim_z + 1,
            output_dim=dim_z,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_norm=False,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate drift network on batched latent states.

        Args:
          t: [] or [N, 1]
          z: [N, dim_z], latent states.

        Returns:
          mu_hat: [N, dim_z], drift output.
        """
        t_col = torch.ones_like(z[..., :1]) * t
        return self.net(torch.cat([t_col, z], dim=-1))


class DiffusionNet(nn.Module):
    """
    Diffusion parameterization for latent SDE.

    Supports:
      - 'const': global learnable scalar times identity (state/time independent)
      - 'diag': per-dimension standard deviation vector
      - 'full': lower-triangular matrix parameterization

    Returned quantity is always a diffusion factor (not covariance):
      - diag: std vector s
      - const: s * I
      - full: lower-triangular L
    and covariance is obtained as factor @ factor^T when needed.
    """

    def __init__(
        self,
        dim_z: int,
        hidden_dim: int | list[int],
        sigma_type: str = "diag",
        dropout: float = 0.0,
    ):
        super().__init__()
        if sigma_type not in ("const", "diag", "full"):
            raise ValueError(f"Unknown sigma_type={sigma_type}")
        self.dim_z = dim_z
        self.sigma_type = sigma_type
        
        self.softplus = nn.Softplus()
        if sigma_type == "const":
            # Single global scalar factor; expanded as scalar * I_D at runtime.
            self.sigma_const_raw = nn.Parameter(torch.randn(1))
        else:
            out_dim = dim_z if sigma_type == "diag" else dim_z * (dim_z + 1) // 2
            self.net = MLP(
                input_dim=dim_z + 1,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                batch_norm=False,
            )

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        """
        Evaluate diffusion parameterization on batched latent states.

        Args:
          t: [] or [N, 1]
          z: [N, dim_z], latent states.

        Returns:
          sigma_factor:
            - [] for `sigma_type='const'` (single global scalar std)
            - [N, dim_z] for `sigma_type='diag'` (positive per-dimension std)
            - [N, dim_z, dim_z] for `sigma_type='full'` (lower-triangular factor)
        """
        if self.sigma_type == "const":
            return self.softplus(self.sigma_const_raw).reshape(()) + STD_MIN_SCALE

        t_col = torch.ones_like(z[..., :1]) * t
        raw = self.net(torch.cat([t_col, z], dim=-1))
        if self.sigma_type == "diag":
            return self.softplus(raw) + STD_MIN_SCALE

        lower = _triangle_vec_to_lower(raw, self.dim_z)
        diag_idx = torch.arange(self.dim_z, device=lower.device)
        lower[..., diag_idx, diag_idx] = self.softplus(lower[..., diag_idx, diag_idx]) + STD_MIN_SCALE
        return lower


class SDESolver(nn.Module):
    """
    Euler-Maruyama latent SDE solver:
      z_{t+1} = z_t + mu_hat_t * dt + B_t * sqrt(dt) * eps_t.

    For sigma_type in {'const', 'full'}, diffusion output is interpreted as a
    Cholesky-like factor L and noise is sampled as L sqrt(dt) eps.
    """

    def __init__(self, drift_net: nn.Module, diffusion_net: nn.Module, sigma_type: str = "diag"):
        super().__init__()
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.sigma_type = sigma_type

    @staticmethod
    def _format_dt(dt: torch.Tensor | float, z_like: torch.Tensor) -> torch.Tensor:
        """
        Normalize time-step shape for Euler-Maruyama updates.

        Args:
          dt: scalar, [B], or [B, 1]
          z_like: [B, dim_z], reference tensor for dtype/device.

        Returns:
          dt_tensor: scalar or [B, 1]
        """
        dt_tensor = torch.as_tensor(dt, device=z_like.device, dtype=z_like.dtype)
        if dt_tensor.ndim == 0:
            return dt_tensor
        if dt_tensor.ndim == 1:
            return dt_tensor.view(-1, 1)
        return dt_tensor

    def get_next_z(
        self,
        z_init: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        include_state_noise: bool = True,
    ):
        """
        Single Euler-Maruyama step.

        Args:
          z_init: [B, dim_z], current latent state batch.
          t: [] or [B, 1], current time.
          dt: [] or [B, 1], time increment.
          include_state_noise: if False, apply drift-only update.
        Returns:
          z_next: [B, dim_z], one-step latent state after Euler-Maruyama update.
        """
        mu_hat = self.drift_net(t, z_init)
        dt_tensor = self._format_dt(dt, z_init)
        if not include_state_noise:
            return z_init + mu_hat * dt_tensor

        sigma_hat = self.diffusion_net(t, z_init)
        sqrt_dt = torch.sqrt(dt_tensor.clamp_min(1.0e-12))

        if self.sigma_type in {"const", "diag"}:
            eps = torch.randn_like(mu_hat) * sqrt_dt
            return z_init + mu_hat * dt_tensor + sigma_hat * eps

        eps = torch.randn(*mu_hat.shape, 1, device=mu_hat.device, dtype=mu_hat.dtype) * sqrt_dt
        return z_init + mu_hat * dt_tensor + torch.matmul(sigma_hat, eps).squeeze(-1)

    def forward(
        self,
        initial_point: torch.Tensor,
        tps: torch.Tensor,
        include_state_noise: bool = True,
    ):
        """
        Roll out latent paths on timeline `tps` [L].

        Args:
          initial_point: [B, dim_z], latent states at the first time point `tps[0]`.
          tps: [L], 1D time grid shared across the batch.
          include_state_noise: if False, use drift-only updates across the rollout.
        Returns:
          sol: [B, L, dim_z], sampled latent SDE trajectory over `tps`.
        """
        if initial_point.ndim != 2:
            raise ValueError(f"SDESolver.forward expects [B, dim_z], got {tuple(initial_point.shape)}")

        batch_size, dim_z = initial_point.shape
        n_tp = tps.shape[0]
        sol = torch.zeros(batch_size, n_tp, dim_z, device=initial_point.device, dtype=initial_point.dtype)
        sol[:, 0, :] = initial_point

        for i in range(1, n_tp):
            dt = tps[i] - tps[i - 1]
            t_prev = tps[i - 1]
            sol[:, i, :] = self.get_next_z(
                sol[:, i - 1, :],
                t_prev,
                dt,
                include_state_noise=include_state_noise,
            )
        return sol


class SDELatentTransition(nn.Module):
    """
    Latent SDE transition block (SDE counterpart of `DiscreteLatentTransition`).

    Owns:
      - drift parameterization f_theta
      - diffusion parameterization g_theta
      - Euler-Maruyama solver for one-step and rollout integration
    """

    def __init__(
        self,
        dim_z: int,
        drift_net_hidden_dim: int | list[int],
        diffusion_net_hidden_dim: int | list[int],
        sigma_type: str = "diag",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_z = dim_z
        self.sigma_type = sigma_type
        self.solver = SDESolver(
            drift_net=DriftNet(dim_z=dim_z, hidden_dim=drift_net_hidden_dim, dropout=dropout),
            diffusion_net=DiffusionNet(
                dim_z=dim_z,
                hidden_dim=diffusion_net_hidden_dim,
                sigma_type=sigma_type,
                dropout=dropout,
            ),
            sigma_type=sigma_type,
        )

    # ---- Private Helpers ----
    def _increment_params(
        self,
        z: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shared increment-parameterization kernel.

        Args:
          z: [N, dim_z]
          t_prev: [N, 1]

        Returns:
          mu: [N, dim_z]
          sigma:
            - [] for `sigma_type='const'`
            - [N, dim_z] for `sigma_type='diag'`
            - [N, dim_z, dim_z] for `sigma_type='full'`
        """
        mu = self.solver.drift_net(t_prev, z)
        sigma = self.solver.diffusion_net(t_prev, z)
        return mu, sigma

    def _apply_em_step(
        self,
        z_prev: torch.Tensor,
        mu_inc_p: torch.Tensor,
        sigma_inc_p: torch.Tensor,
        delta_t: torch.Tensor,
        include_state_noise: bool,
    ) -> torch.Tensor:
        """
        Apply one Euler-Maruyama update from parameterized increment terms.

        Args:
          z_prev: [*, dim_z]
          mu_inc_p: [*, dim_z]
          sigma_inc_p: [] (const), [*, dim_z] (diag), or [*, dim_z, dim_z] (full)
          delta_t: scalar or [*, 1]
          include_state_noise: whether to include stochastic diffusion term.

        Returns:
          z_next: [*, dim_z]
        """
        dt = delta_t
        drift_term = mu_inc_p * dt
        if not include_state_noise:
            return z_prev + drift_term

        sqrt_dt = torch.sqrt(dt.clamp_min(1.0e-12))
        if self.sigma_type in {"const", "diag"}:
            eps = torch.randn_like(mu_inc_p) * sqrt_dt
            return z_prev + drift_term + sigma_inc_p * eps

        eps = torch.randn(*mu_inc_p.shape, 1, device=mu_inc_p.device, dtype=mu_inc_p.dtype) * sqrt_dt
        noise_term = torch.matmul(sigma_inc_p, eps).squeeze(-1)
        return z_prev + drift_term + noise_term

    # ---- Public Transition APIs ----
    def forward(
        self,
        z_prev: torch.Tensor,
        t_prev: torch.Tensor,
        delta_t: torch.Tensor,
        include_state_noise: bool = True,
    ) -> torch.Tensor:
        """
        Compute one latent SDE step (drift-only or stochastic Euler-Maruyama).

        Args:
          z_prev: [B, dim_z], latent state at current step.
          t_prev: [B, 1], current time.
          delta_t: [B, 1], step size to next time.
          include_state_noise: if False, use drift-only update.

        Returns:
          z_next: [B, dim_z], one-step latent transition result.
        """
        mu_inc_p, sigma_inc_p = self._increment_params(z_prev, t_prev)
        return self._apply_em_step(
            z_prev=z_prev,
            mu_inc_p=mu_inc_p,
            sigma_inc_p=sigma_inc_p,
            delta_t=delta_t,
            include_state_noise=include_state_noise,
        )

    def compute_increment_params(
        self,
        z_in: torch.Tensor,
        tps_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute latent SDE increment parameterization terms on a latent anchor trajectory.

        Args:
          z_in: [B, L-1, dim_z], latent inputs used to parameterize increments.
          tps_in: [B, L-1], current-step timestamps aligned with `z_in`.

        Returns:
          mu_inc_p: [B, L-1, dim_z], drift means for each increment.
          sigma_inc_p:
            - [] for `sigma_type='const'` (single global scalar std)
            - [B, L-1, dim_z] for `sigma_type='diag'`
            - [B, L-1, dim_z, dim_z] for `sigma_type='full'`

        Notes:
          This function returns raw transition parameters on anchor states:
            mu_inc_p = f_theta(t_{k-1}, z_{k-1}^{in})
          sigma_inc_p = g_theta(t_{k-1}, z_{k-1}^{in})
          It does not apply delta-t scaling. `delta_t` is prepared by the caller.
        """
        z = z_in.reshape(-1, self.dim_z)
        t_prev = tps_in.reshape(-1, 1)
        mu, sigma = self._increment_params(z, t_prev)

        mu_inc_p = mu.view(z_in.shape[0], z_in.shape[1], self.dim_z)
        if self.sigma_type == "const":
            sigma_inc_p = sigma
        elif self.sigma_type == "diag":
            sigma_inc_p = sigma.view(z_in.shape[0], z_in.shape[1], self.dim_z)
        else:
            sigma_inc_p = sigma.view(z_in.shape[0], z_in.shape[1], self.dim_z, self.dim_z)
        return mu_inc_p, sigma_inc_p

    def one_step_transition(
        self, 
        z_in: torch.Tensor,
        tps_in: torch.Tensor,
        delta_t: torch.Tensor,
        include_state_noise: bool = True) -> torch.Tensor:
        """
        Roll one-step latent transitions from pre-aligned increment inputs.

        Args:
          z_in: [B, L-1, dim_z], latent inputs used as one-step anchors.
          tps_in: [B, L-1], current-step timestamps aligned with `z_in`.
          delta_t: [B, L-1, 1], step sizes aligned with `z_in`.
          include_state_noise: if False, use drift-only update.

        Returns:
          z_plus: [B, L-1, dim_z], one-step latent predictions for times `t_1..t_{L-1}`.
        """
        mu_inc_p, sigma_inc_p = self.compute_increment_params(z_in, tps_in)
        z_plus = self._apply_em_step(
            z_prev=z_in,
            mu_inc_p=mu_inc_p,
            sigma_inc_p=sigma_inc_p,
            delta_t=delta_t,
            include_state_noise=include_state_noise,
        )
        return z_plus

    def multi_step_rollout(
        self, 
        initial_point: torch.Tensor, 
        tps: torch.Tensor, 
        include_state_noise: bool = True) -> torch.Tensor:
        """
        Roll out latent SDE trajectories on per-batch timelines.

        Args:
          initial_point: [B, dim_z]
          tps: [B, L], per-batch rollout time grid (absolute times).
          include_state_noise: if False, use drift-only updates across all rollout steps.

        Returns:
          sol: [B, L, dim_z]
        """
        if initial_point.ndim != 2:
            raise ValueError(f"Expected initial_point [B, dim_z], got {tuple(initial_point.shape)}")
        if tps.ndim != 2:
            raise ValueError(f"Expected tps [B, L], got {tuple(tps.shape)}")
        
        batch_size, dim_z = initial_point.shape
        n_steps = tps.shape[1]
        
        sol = torch.zeros(batch_size, n_steps, dim_z, device=initial_point.device, dtype=initial_point.dtype)
        sol[:, 0, :] = initial_point
        
        for step_idx in range(1, n_steps):
            delta_t = (tps[:, step_idx] - tps[:, step_idx - 1]).unsqueeze(-1)  # [B, 1]
            t_prev = tps[:, step_idx - 1].unsqueeze(-1)  # [B, 1]
            sol[:, step_idx, :] = self(
                z_prev=sol[:, step_idx - 1, :],
                t_prev=t_prev,
                delta_t=delta_t,
                include_state_noise=include_state_noise,
            )
        return sol


class LatentSDEModel(nn.Module):
    """
    Latent Neural SDE model with ident-latent-sde style training hooks.

    Components:
      - posterior encoder q(z_t|x)
      - latent transition block (drift + diffusion + solver)
      - observation decoder p(x_t|z_t)

    This module exposes high-level posterior sampling, reconstruction,
    and forecasting APIs. Loss-specific tensor routing is handled in SDESSM.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        enc_rnn_hidden_dim: int,
        enc_rnn_num_layers: int,
        enc_proj_dim: int,
        drift_net_hidden_dim: int | list[int],
        diffusion_net_hidden_dim: int | list[int],
        dec_hidden_dim: int | list[int],
        x_sigma2: float = 0.01,
        z_sigma_type: str = "diag",
        enc_direction: str = "backward",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sigma_type = z_sigma_type
        self.enc_direction = enc_direction

        self.posterior_encoder = RNNEncoder(
            dim_x=dim_x,
            dim_z=dim_z,
            rnn_hidden_dim=enc_rnn_hidden_dim,
            rnn_num_layers=enc_rnn_num_layers,
            proj_dim=enc_proj_dim,
            sigma_type=z_sigma_type,
            dropout=dropout,
        )
        self.latent_transition = SDELatentTransition(
            dim_z=dim_z,
            drift_net_hidden_dim=drift_net_hidden_dim,
            diffusion_net_hidden_dim=diffusion_net_hidden_dim,
            sigma_type=z_sigma_type,
            dropout=dropout,
        )
        self.obs_decoder = GaussianObsDecoder(
            dim_z=dim_z,
            dim_x=dim_x,
            hidden_dim=dec_hidden_dim,
            x_sigma2=x_sigma2,
            dropout=dropout,
            batch_norm=False,
        )

    # ---- Public Training APIs ----
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        mc_samples: int = 0,
    ) -> tuple:
        """
        Alias of `reconstruct` (plus-one training-style reconstruction path).

        Args:
          x: [B, L, dim_x]
          mask: [B, L, dim_x] or None
          time_steps: [B, L]
          mc_samples: number of posterior samples to average for transition anchor.
            - 0: use posterior mean anchor (mu_zq)
            - >=1: use averaged sampled posterior anchor
        """
        return self.reconstruct(
            x=x,
            mask=mask,
            time_steps=time_steps,
            mc_samples=mc_samples,
        )

    def sample_posterior_latent_path(
        self,
        q_mu: torch.Tensor,
        q_sigma: torch.Tensor,
        mc_samples: int = 1,
        take_mc_average: bool = True,
    ):
        """
        Sample a latent trajectory from the posterior and compute its increments.

        Diagonal case:
          z_t = q_mu_t + sigma_t * eps_t
        Const case:
          z_t = q_mu_t + sigma_const * eps_t
        Full-covariance case:
          z_t = q_mu_t + (L_t L_t^T) eps_t

        Increments:
          inc_t = z_t - z_{t-1}

        Args:
          q_mu: [B, L, dim_z], posterior mean trajectory.
          q_sigma: [] (const scalar std), [B, L, dim_z] (diag), or [B, L, dim_z*(dim_z+1)//2] (packed Cholesky).
          mc_samples: number of posterior path samples (must be >= 1).
          take_mc_average: if True, average the MC samples along the sample dimension; otherwise return all samples.

        Returns:
          sampled_z:
            - [S, B, L, dim_z] if `take_mc_average=False`
            - [B, L, dim_z] if `take_mc_average=True`
        """
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1 for posterior sampling, got {mc_samples}.")

        if self.sigma_type == "diag":
            if q_sigma.shape != q_mu.shape:
                raise ValueError(
                    f"Expected q_sigma shape {q_mu.shape} for diag posterior, got {q_sigma.shape}."
                )
            # Diagonal posterior: z_t = q_mu_t + q_sigma_t * eps_t.
            sampled_z_mc = Sampler.sample_normal(
                q_mu,
                q_sigma ** 2,
                mc_samples=mc_samples,
                take_mc_average=take_mc_average,
            )  
        elif self.sigma_type == "const":
            if q_sigma.ndim != 0:
                raise ValueError(f"Expected scalar q_sigma for const posterior, got shape {q_sigma.shape}.")
            # Constant posterior: single global scalar std shared across batches/times/dimensions.
            q_sigma_full_like = q_sigma.reshape(1, 1, 1).expand_as(q_mu)
            sampled_z_mc = Sampler.sample_normal(
                q_mu,
                q_sigma_full_like ** 2,
                mc_samples=mc_samples,
                take_mc_average=take_mc_average,
            )
        elif self.sigma_type == "full":
            tri_dim = self.dim_z * (self.dim_z + 1) // 2
            if q_sigma.shape[-1] != tri_dim:
                raise ValueError(
                    f"Expected q_sigma last dim {tri_dim} for full posterior, got {q_sigma.shape[-1]}."
                )
            # Full posterior: z_t = q_mu_t + L_t eps_t, with cov_t = L_t L_t^T.
            lower_triangular_q = _triangle_vec_to_lower(q_sigma, self.dim_z)
            sampled_z_mc = Sampler.sample_normal_full(
                q_mu,
                lower_triangular_q,
                mc_samples=mc_samples,
                take_mc_average=take_mc_average,
                is_cholesky=True,
            )  
        else:
            raise ValueError(f"Unsupported sigma_type={self.sigma_type}")

        return sampled_z_mc

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        mc_samples: int = 0
    ) -> tuple:
        """
        Build the plus-one SDE reconstruction path used by SDESSM training.

        Args:
          x: [B, L, dim_x], context observations.
          mask: [B, L, dim_x], context observation mask.
          time_steps: [B, L], context timestamps aligned with `x`.
            Uses model-level `self.enc_direction`.
          mc_samples: posterior sample count used for transition anchors.

        This method:
          1) encodes posterior parameters on the observed context,
          2) optionally samples posterior latent paths and averages them,
          3) computes SDE increment statistics and plus-one latent transitions,
          4) decodes the plus-one latent path for reconstruction,
          5) returns reconstruction tensors plus posterior latent terms.

        Returns:
          x_plus_mean: [B, L-1, dim_x], plus-one decoded means used for reconstruction loss.
          x_plus_sigma2: [B, L-1, dim_x], fixed observation variance tensor.
          mu_zq: [B, L, dim_z], posterior means (unexpanded).
          sigma_zq: [] (const scalar std), [B, L, dim_z] (diag), or [B, L, tri_dim] (full packed factor).
          sampled_z: [B, L, dim_z], sampled posterior latent path used to build increment observations.

        Notes:
          - targets/masks are retrieved from the training batch in `SDESSM._step`.
          - KL routing and increment-parameter calculation are handled in `SDESSM._step`.
          - include_state_noise=False makes the plus-one reconstruction path drift-only.
        """
        # Stage 1: posterior encoding on the observed context.
        mu_zq, sigma_zq = self.posterior_encoder(
            x, mask, time_steps, encoding_direction=self.enc_direction
        )  # [B, L, dim_z], [B, L, ...]

        # Stage 2: posterior sampling and transition-anchor selection.
        # Always return an actually sampled latent path for downstream losses.
        sampled_z = self.sample_posterior_latent_path(
            mu_zq,
            sigma_zq,
            mc_samples=max(mc_samples, 1),
            take_mc_average=True,
        )  # [B, L, dim_z]
        
        if mc_samples == 0:
            transition_anchor = mu_zq
        else:  # mc_samples >= 1
            transition_anchor = sampled_z     # [B, L, dim_z] (stabilized sampled anchor)

        # Stage 3: plus-one latent rollout from selected transition anchor.
        z_in = transition_anchor[:, :-1, :]  # [B, L-1, dim_z]
        tps_in = time_steps[:, :-1]  # [B, L-1]
        delta_t = (time_steps[:, 1:] - time_steps[:, :-1]).unsqueeze(-1)  # [B, L-1, 1]
        z_plus = self.latent_transition.one_step_transition(
            z_in=z_in,
            tps_in=tps_in,
            delta_t=delta_t,
            include_state_noise=True,
        )  # [B, L-1, dim_z]

        # Stage 4: decode plus-one latent paths for reconstruction.
        x_plus_mean, x_plus_sigma2 = self.obs_decoder(z_plus)  # [B, L-1, dim_x]

        return (
            x_plus_mean,
            x_plus_sigma2,
            mu_zq,
            sigma_zq,
            sampled_z,
        )

    @torch.no_grad()
    def smooth(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple:
        """
        Compute smoothed posterior estimates q(z_t | x_{1:T}) for all t.
        
        This runs the encoder to obtain smoothed latent estimates using observations
        (akin to RTS/Kalman smoothing). Direction is configured at model level
        via `self.enc_direction`.
        
        Args:
          x: [B, L, dim_x], observed trajectory.
          mask: [B, L, dim_x], observation mask.
          time_steps: [B, L], timestamps aligned with `x`.
        
        Returns:
          mu_zq: [B, L, dim_z], smoothed posterior latent means.
          sigma_zq: [] (const), [B, L, dim_z] (diag), or [B, L, tri_dim] (full),
                    smoothed posterior latent scale parameters.
        """
        mu_zq, sigma_zq = self.posterior_encoder(
            x, mask, time_steps, encoding_direction=self.enc_direction
        )
        return mu_zq, sigma_zq

    # ---- Public Forecast APIs ----
    @torch.no_grad()
    def _forecast_from_state_batched(
        self,
        initial_state: torch.Tensor,
        tps: torch.Tensor,
        n_trajs: int = 1,
    ):
        """
        Helper forecast from normalized batched latent anchor.

        Args:
          initial_state: [B, dim_z]
          tps: [B, L_pred+1], absolute prediction timeline (includes t_init as first element).
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (drift-only latent path, observation mean only).
            - >1: stochastic rollout with latent diffusion noise and sampled observations.

        Returns:
          x_samples:
            - [B, L_pred, dim_x] when n_trajs == 1
            - [S, B, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [B, L_pred, dim_z] when n_trajs == 1
            - [S, B, L_pred, dim_z] when n_trajs > 1
        """
        if n_trajs < 1:
            raise ValueError(f"Expected n_trajs >= 1, got {n_trajs}")
        if n_trajs == 1:
            z_path = self.latent_transition.multi_step_rollout(
                initial_point=initial_state,
                tps=tps,
                include_state_noise=False,
            )  # [B, L_pred+1, dim_z]
            z_samples = z_path[:, 1:, :]  # [B, L_pred, dim_z]
            x_mean, _ = self.obs_decoder(z_samples)
            return x_mean, z_samples

        batch_size = initial_state.shape[0]
        z0_batch = initial_state.unsqueeze(0).repeat(n_trajs, 1, 1).reshape(
            n_trajs * batch_size, self.dim_z
        )  # [S*B, dim_z]
        tps_batch = tps.unsqueeze(0).expand(n_trajs, batch_size, tps.shape[1]).reshape(
            n_trajs * batch_size, tps.shape[1]
        )  # [S*B, L_pred+1]

        z_path_flat = self.latent_transition.multi_step_rollout(
            initial_point=z0_batch,
            tps=tps_batch,
            include_state_noise=True,
        )  # [S*B, L_pred+1, dim_z]
        z_path_mc = z_path_flat.view(
            n_trajs, batch_size, z_path_flat.shape[1], self.dim_z
        )  # [S, B, L_pred+1, dim_z]
        z_samples = z_path_mc[:, :, 1:, :]  # [S, B, L_pred, dim_z]
        x_mean, x_sigma2 = self.obs_decoder(z_samples)
        x_samples = Sampler.sample_normal(
            mu=x_mean,
            var=x_sigma2,
            mc_samples=1,
            take_mc_average=False,
        ).squeeze(0)  # [S, B, L_pred, dim_x]

        return x_samples, z_samples
    
    @torch.no_grad()
    def forecast_from_state(
        self,
        initial_state: torch.Tensor,
        tps: torch.Tensor,
        n_trajs: int = 1,
    ) -> tuple:
        """
        Helper forecast from unbatched latent anchor.

        Args:
          initial_state: [dim_z]
          tps: [L_pred+1], absolute prediction timeline (includes t_init as first element).
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (drift-only latent path, observation mean only).
            - >1: stochastic rollout with latent diffusion noise and sampled observations.

        Returns:
          x_samples:
            - [L_pred, dim_x] when n_trajs == 1
            - [S, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [L_pred, dim_z] when n_trajs == 1
            - [S, L_pred, dim_z] when n_trajs > 1
        """
        # Normalize inputs to have batch dimension
        initial_state = initial_state.unsqueeze(0)  # [1, dim_z]
        tps = tps.unsqueeze(0)  # [1, L_pred+1]
        
        x_samples, z_samples = self._forecast_from_state_batched(
            initial_state=initial_state,
            tps=tps,
            n_trajs=n_trajs,
        )
        
        if n_trajs == 1:
            x_samples = x_samples.squeeze(0)  # [L_pred, dim_x]
            z_samples = z_samples.squeeze(0)  # [L_pred, dim_z]
        else:
            x_samples = x_samples.squeeze(1)  # [S, L_pred, dim_x]
            z_samples = z_samples.squeeze(1)  # [S, L_pred, dim_z]
        
        return x_samples, z_samples

    @torch.no_grad()
    def forecast_from_context(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_predict: torch.Tensor,
        n_trajs: int = 1,
    ) -> tuple:
        """
        Forecast from unbatched observed context.

        Args:
          x: [L_obs, dim_x]
          mask: [L_obs, dim_x]
          time_steps: [L_obs]
          time_steps_to_predict: [L_pred]
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (drift-only latent path, observation mean only).
            - >1: stochastic rollout with latent diffusion noise and sampled observations.

        Returns:
          x_samples:
            - [L_pred, dim_x] when n_trajs == 1
            - [S, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [L_pred, dim_z] when n_trajs == 1
            - [S, L_pred, dim_z] when n_trajs > 1
        """
        # Stage 1: Batch inputs and infer last latent state
        x_batched = x.unsqueeze(0)  # [1, L_obs, dim_x]
        time_steps_batched = time_steps.unsqueeze(0)  # [1, L_obs]
        mask_batched = mask.unsqueeze(0)  # [1, L_obs, dim_x]
        
        mu_zq, _ = self.smooth(
            x_batched, mask_batched, time_steps_batched
        )
        initial_state = mu_zq[0, -1]  # [dim_z]
        t_init = time_steps[-1]  # scalar
        
        # Stage 2: Build absolute prediction timeline starting from t_init
        tps_forecast = torch.cat([t_init.unsqueeze(0), time_steps_to_predict], dim=0)  # [L_pred+1]

        return self.forecast_from_state(
            initial_state=initial_state,
            tps=tps_forecast,
            n_trajs=n_trajs,
        )
