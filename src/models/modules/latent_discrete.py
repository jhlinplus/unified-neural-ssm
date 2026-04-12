import torch
import torch.nn as nn

from .common import MLP, Sampler, GaussianObsDecoder, STD_MIN_SCALE

##############################
# Non-masked modules 
##############################
class BiRNNEncoder(nn.Module):
    """
    Sequence encoder for discrete latent dynamics.

    Maps observations x_{1:T} to per-step posterior parameters q(z_t | x_{1:T})
    using a projected bidirectional GRU:
      x_t -> h_t -> (mu_t, sigma_t).
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        x_proj_dim: int,
        rnn_hidden_dim: int = 64,
        rnn_num_layers: int = 1,
        mask_processor_kernel_size: int = 3,
        dropout: float = 0.1,
        sigma_type: str = "const",
    ) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.sigma_type = sigma_type
        self.softplus = nn.Softplus()
        if self.sigma_type not in {"const", "diag"}:
            raise ValueError(f"Unsupported sigma_type={self.sigma_type}. Expected one of {{'const','diag'}}.")

        self.x_projection = MLP(
            input_dim=dim_x,
            output_dim=x_proj_dim,
            hidden_dim=[x_proj_dim],
            dropout=dropout,
            batch_norm=False,
        )
        
        self.mask_1dcnn = nn.Conv1d(dim_x, 2 * x_proj_dim, mask_processor_kernel_size, padding="same")
        self.mask_projector = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * x_proj_dim, x_proj_dim),
        )

        self.birnn = nn.GRU(
            input_size=x_proj_dim + 1,
            hidden_size=self.rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.combiner_forward_mu = nn.Linear(rnn_hidden_dim, self.dim_z)
        self.combiner_backward_mu = nn.Linear(rnn_hidden_dim, self.dim_z)
        if self.sigma_type == "const":
            self.sigma_const_raw = nn.Parameter(torch.randn(1))
        else:
            self.combiner_forward_sigma = nn.Sequential(
                nn.Linear(rnn_hidden_dim, self.dim_z),
                nn.Softplus(),
            )
            self.combiner_backward_sigma = nn.Sequential(
                nn.Linear(rnn_hidden_dim, self.dim_z),
                nn.Softplus(),
            )

    @staticmethod
    def merge(mu1, sigma2_1, mu2, sigma2_2):
        """Product-of-Gaussians-style merge for forward/backward Gaussian messages."""
        mu = (mu2 * sigma2_1 + mu1 * sigma2_2) / (sigma2_1 + sigma2_2)
        sigma2 = sigma2_2 * sigma2_1 / (sigma2_1 + sigma2_2)
        return mu, sigma2

    @staticmethod
    def get_delta_t(tps):
        """Build per-step delta time with zero padding at the first step."""
        delta_t = tps[:, 1:] - tps[:, :-1]
        delta_t = torch.cat([torch.zeros(tps.shape[0], 1, device=delta_t.device), delta_t], dim=1)
        return delta_t.unsqueeze(-1)

    def _project_x(self, x):
        return self.x_projection(x)

    def _build_h0(self, batch_size, device, h0_forward=None, h0_backward=None):
        if h0_forward is None:
            h0_forward = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        if h0_backward is None:
            h0_backward = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
        return torch.cat([h0_forward, h0_backward], dim=0)

    def forward(self, x, mask, tps, h0_forward=None, h0_backward=None):
        """
        Args:
          x:   [B, L, dim_x]
          mask: [B, L, dim_x], observation-availability mask.
          tps: [B, L]
        Returns:
          mu: [B, L, dim_z], merged posterior mean per time step.
          sigma2: [B, L, dim_z], posterior variance per time step.
          h_l: [B, L, rnn_hidden_dim], forward GRU hidden states.
          h_r: [B, L, rnn_hidden_dim], backward GRU hidden states.
        """
        if x.shape[-1] != self.dim_x:
            raise ValueError(f"x.shape={x.shape}; expecting last dim {self.dim_x}")
        if x.shape != mask.shape:
            raise ValueError(f"x.shape={x.shape}; mask.shape={mask.shape}; expecting same shape")
        h0 = self._build_h0(x.shape[0], x.device, h0_forward, h0_backward)
        
        hx = self._project_x(x)
        mask_hidden = self.mask_projector(self.mask_1dcnn(mask.permute(0, 2, 1)).permute(0, 2, 1))
        
        hx = hx + mask_hidden
        delta_t = self.get_delta_t(tps)
        hx = torch.cat([hx, delta_t], dim=-1)

        h, _ = self.birnn(hx, h0)
        h_l, h_r = h[..., :self.rnn_hidden_dim], h[..., self.rnn_hidden_dim :]

        mu_l = self.combiner_forward_mu(h_l)
        mu_r = self.combiner_backward_mu(h_r)
        if self.sigma_type == "const":
            mu = 0.5 * (mu_l + mu_r)
            sigma_std_scalar = self.softplus(self.sigma_const_raw).to(mu) + STD_MIN_SCALE
            sigma2 = (sigma_std_scalar ** 2).view(1, 1, 1).expand_as(mu)
        else:
            sigma_l = self.combiner_forward_sigma(h_l) + STD_MIN_SCALE
            sigma_r = self.combiner_backward_sigma(h_r) + STD_MIN_SCALE
            sigma2_l = sigma_l ** 2
            sigma2_r = sigma_r ** 2
            mu, sigma2 = self.merge(mu_l, sigma2_l, mu_r, sigma2_r)
        return mu, sigma2, h_l, h_r


class ArTransition(nn.Module):
    """
    Discrete autoregressive latent transition block.

    Parameterizes:
      p(z_t | z_{t-1}, dt_t) via mean net `f_mean` and std net/parameter.

    Std is parameterized (const or diag), and variance is formed as sigma^2
    before being returned to loss/sampling callers.
    """

    def __init__(
        self,
        dim_z: int,
        proj_dim: int,
        mean_net_hidden_dim: int | list[int] | None = None,
        sigma_net_hidden_dim: int | list[int] | None = None,
        sigma_type: str = "const",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim_z = dim_z
        self.proj_dim = proj_dim
        if sigma_type not in {"const", "diag"}:
            raise ValueError(f"Unsupported sigma_type={sigma_type}. Expected one of {{'const','diag'}}.")
        self.sigma_type = sigma_type
        self.softplus = nn.Softplus()
        
        self.z_projection = MLP(
            input_dim=dim_z,
            output_dim=proj_dim,
            hidden_dim=[proj_dim],
            dropout=dropout,
            batch_norm=False,
        )

        self.f_mean = MLP(
            input_dim=proj_dim + 1,
            output_dim=self.dim_z,
            hidden_dim=mean_net_hidden_dim,
            dropout=dropout,
            batch_norm=False,
        )

        if self.sigma_type == "const":
            self.sigma_const_raw = nn.Parameter(torch.randn(1))
        else:
            self.f_sigma = MLP(
                input_dim=proj_dim + 1,
                output_dim=self.dim_z,
                hidden_dim=sigma_net_hidden_dim,
                dropout=dropout,
                batch_norm=False,
            )

    # ---- Private Helpers ----
    def _step_params(self, z_inputs: torch.Tensor, dt_inputs: torch.Tensor):
        """
        Private per-step parameterization kernel on flattened inputs.

        Args:
          z_inputs: [N, dim_z]
          dt_inputs: [N, 1]
        Returns:
          z_next_mean: [N, dim_z]
          z_next_sigma2: [N, dim_z], variance built from std parameterization.
        """
        z_projected = self.z_projection(z_inputs)
        z_projected_with_t = torch.cat([z_projected, dt_inputs], dim=-1)
        z_next_mean = self.f_mean(z_projected_with_t)
        if self.sigma_type == "const":
            z_next_std_scalar = self.softplus(self.sigma_const_raw).to(z_next_mean) + STD_MIN_SCALE
            z_next_std = z_next_std_scalar.view(1, 1).expand(z_next_mean.shape[0], self.dim_z)
        else:
            z_next_std = self.softplus(self.f_sigma(z_projected_with_t)) + STD_MIN_SCALE
        z_next_sigma2 = z_next_std ** 2
        return z_next_mean, z_next_sigma2

    # ---- Public Transition APIs ----
    def forward(
        self,
        z_prev,
        delta_t,
        include_state_noise=True,
    ):
        """
        Compute one discrete latent transition step (counterpart to SDE transition forward).

        Args:
          z_prev: [B, dim_z]
          delta_t: [B, 1]
          include_state_noise: whether to sample transition noise.

        Returns:
          z_curr: [B, dim_z], transitioned latent state.
        """
        z_curr, z_curr_sigma2 = self._step_params(z_prev, delta_t)
        if include_state_noise:
            z_curr = Sampler.sample_normal(z_curr, z_curr_sigma2, mc_samples=1)
        return z_curr

    def compute_next_step_params(self, z_in: torch.Tensor, delta_t: torch.Tensor):
        """
        Compute one-step transition parameters from pre-aligned transition inputs.

        This is the public sequence-level wrapper around `_step_params(...)`.

        Args:
          z_in: [B_like, T-1, dim_z], latent anchors z_{1:T-1}.
          delta_t: [B_like, T-1, 1], per-step time increments aligned with `z_in`.

        Returns:
          z_next_mean: [B_like, T-1, dim_z], one-step transition means for z_{2:T}.
          z_next_sigma2: [B_like, T-1, dim_z], one-step transition variances for z_{2:T}.
        """
        ## flatten batch and time dimensions for vectorized computation
        flat_z_prev = z_in.reshape(-1, self.dim_z)
        flat_dt = delta_t.reshape(-1, 1)
        z_next_mean_flat, z_next_sigma2_flat = self._step_params(flat_z_prev, flat_dt)
        z_next_mean = z_next_mean_flat.view_as(z_in)
        z_next_sigma2 = z_next_sigma2_flat.view_as(z_in)
        return z_next_mean, z_next_sigma2

    def one_step_transition(self, z_in, delta_t, include_state_noise=True):
        """
        Apply one-step transitions over pre-aligned latent inputs.

        Args:
          z_in: [B, L-1, dim_z]
          delta_t: [B, L-1, 1]
          include_state_noise: whether to sample transition noise.

        Returns:
          z_plus: [B, L-1, dim_z]
        """
        z_next_mean, z_next_sigma2 = self.compute_next_step_params(
            z_in=z_in,
            delta_t=delta_t,
        )
        if include_state_noise:
            return Sampler.sample_normal(z_next_mean, z_next_sigma2, mc_samples=1)
        return z_next_mean

    def multi_step_rollout(self, initial_point, delta_tps, include_state_noise=True):
        """
        Roll out latent trajectories given per-step time increments `delta_tps`.

        Args:
          initial_point: [B, dim_z]
          delta_tps: [B, L]
          include_state_noise: whether to sample transition noise.

        Returns:
          sol: [B, L+1, dim_z]
        """
        n_batch, dim_z = initial_point.shape
        n_steps = delta_tps.shape[1]
        
        sol = torch.zeros(n_batch, n_steps + 1, dim_z, device=initial_point.device, dtype=initial_point.dtype)
        sol[:, 0, :] = initial_point
        
        for step_idx in range(n_steps):
            delta_t = delta_tps[:, step_idx].unsqueeze(-1)
            sol[:, step_idx + 1, :] = self(
                z_prev=sol[:, step_idx, :],
                delta_t=delta_t,
                include_state_noise=include_state_noise,
            )
        return sol


class LatentDiscreteModel(nn.Module):
    """
    Clean discrete latent state-space model.

    End-to-end:
      x -> q(z_{1:L}|x) -> sample z -> p(z_t|z_{t-1}) -> p(x_t|z_t)

    Shape convention:
      x: [B, L, dim_x]
      time_steps: [B, L]
      latent states: [B, L, dim_z]
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        enc_proj_dim: int,
        enc_rnn_hidden_dim: int,
        enc_rnn_num_layers: int,
        ar_proj_hidden_dim: int,
        ar_mean_net_hidden_dim: int | list[int],
        ar_sigma_net_hidden_dim: int | list[int],
        dec_hidden_dim: int | list[int],
        mask_processor_kernel_size: int,
        z_sigma_type: str = "const",
        x_sigma2: float = 0.01,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.posterior_encoder = BiRNNEncoder(
            dim_x=dim_x,
            dim_z=dim_z,
            x_proj_dim=enc_proj_dim,
            rnn_hidden_dim=enc_rnn_hidden_dim,
            rnn_num_layers=enc_rnn_num_layers,
            sigma_type=z_sigma_type,
            mask_processor_kernel_size=mask_processor_kernel_size,
            dropout=dropout,
        )
        self.latent_transition = ArTransition(
            dim_z=dim_z,
            proj_dim=ar_proj_hidden_dim,
            mean_net_hidden_dim=ar_mean_net_hidden_dim,
            sigma_type=z_sigma_type,
            sigma_net_hidden_dim=ar_sigma_net_hidden_dim,
            dropout=dropout,
        )
        self.obs_decoder = GaussianObsDecoder(
            dim_z=dim_z,
            hidden_dim=dec_hidden_dim,
            dim_x=dim_x,
            x_sigma2=x_sigma2,
            dropout=dropout,
            batch_norm=False,
        )

    # ---- Public Training APIs ----
    def forward(self, x, mask, time_steps, mc_samples=1):
        """
        Alias of `reconstruct`.

        Args:
          x: [B, L, dim_x]
          mask: [B, L, dim_x]
          time_steps: [B, L]
          mc_samples: int
        """
        return self.reconstruct(x=x, mask=mask, time_steps=time_steps, mc_samples=mc_samples)

    def reconstruct(self, x, mask, time_steps, mc_samples=1):
        """
        Build plus-one reconstruction tensors for training.

        Args:
          x: [B, L, dim_x]
          mask: [B, L, dim_x], observation-availability mask for encoder.
          time_steps: [B, L]
          mc_samples: number of Monte Carlo posterior samples before MC averaging.

        Returns:
          x_plus_mean: [B, L-1, dim_x], plus-one decoded means.
          x_plus_sigma2: [B, L-1, dim_x], plus-one decoded variances.
          mu_zq: [B, L, dim_z], posterior latent means.
          sigma2_zq: [B, L, dim_z], posterior latent variances.
          mu_zp_plus: [B, L-1, dim_z], one-step transition means.
          sigma2_zp_plus: [B, L-1, dim_z], one-step transition variances.

        Notes:
          - Posterior samples are averaged across MC draws to build a single latent path [B, L, dim_z] before plus-one transition decoding.
          - When `mc_samples` is smaller than 1, the posterior mean is used as the transition anchor.
        """
        mu_zq, sigma2_zq, _, _ = self.posterior_encoder(x, mask, time_steps)
        delta_t = (time_steps[:, 1:] - time_steps[:, :-1]).unsqueeze(-1)

        if mc_samples < 1:
            transition_anchor = mu_zq
        else:
            transition_anchor = Sampler.sample_normal(mu_zq, sigma2_zq, mc_samples=mc_samples, take_mc_average=True)

        z_in = transition_anchor[:, :-1, :]
        mu_zp_plus, sigma2_zp_plus = self.latent_transition.compute_next_step_params(
            z_in=z_in,
            delta_t=delta_t,
        )
        x_plus_mean, x_plus_sigma2 = self.obs_decoder(mu_zp_plus)
        
        return x_plus_mean, x_plus_sigma2, mu_zq, sigma2_zq, mu_zp_plus, sigma2_zp_plus

    @torch.no_grad()
    def smooth(self, x: torch.Tensor, mask: torch.Tensor, time_steps: torch.Tensor) -> tuple:
        """
        Compute smoothed posterior estimates q(z_t | x_{1:T}) for all t.
        
        This runs the bidirectional encoder to obtain smoothed latent estimates
        using both past and future observations (akin to RTS/Kalman smoothing).
        
        Args:
          x: [B, L, dim_x], observed trajectory.
          mask: [B, L, dim_x], observation-availability mask.
          time_steps: [B, L], timestamps aligned with `x`.
        
        Returns:
          mu_zq: [B, L, dim_z], smoothed posterior latent means.
          sigma2_zq: [B, L, dim_z], smoothed posterior latent variances.
        """
        mu_zq, sigma2_zq, _, _ = self.posterior_encoder(x, mask, time_steps)
        return mu_zq, sigma2_zq

    # ---- Public Forecast APIs ----
    @torch.no_grad()
    def _forecast_from_state_batched(
        self,
        initial_state,
        delta_tps,
        n_trajs: int = 1,
    ):
        """
        Helper forecast from normalized batched latent anchor.

        Args:
          initial_state: [B, dim_z]
          delta_tps: [B, L_pred]
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (no state noise, observation mean only).
            - >1: stochastic rollout with both state and observation noise.

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
        batch_size = initial_state.shape[0]

        # Deterministic single rollout: transition mean path + observation mean path.
        if n_trajs == 1:
            z_path = self.latent_transition.multi_step_rollout(
                initial_point=initial_state,
                delta_tps=delta_tps,
                include_state_noise=False,
            )  # [B, L_pred+1, dim_z]
            z_samples = z_path[:, 1:, :]  # [B, L_pred, dim_z]
            x_mean, _ = self.obs_decoder(z_samples)
            return x_mean, z_samples

        # Stochastic multi-rollout: include both latent transition and observation noise.
        z0_batch = initial_state.unsqueeze(0).repeat(n_trajs, 1, 1).reshape(n_trajs * batch_size, self.dim_z)  # [S, B, dim_z] -> [S*B, dim_z]
        delta_tps_batch = delta_tps.unsqueeze(0).expand(n_trajs, batch_size, delta_tps.shape[1]).reshape(n_trajs * batch_size, delta_tps.shape[1])  # [S*B, L_pred]

        z_path_flat = self.latent_transition.multi_step_rollout(
            initial_point=z0_batch,
            delta_tps=delta_tps_batch,
            include_state_noise=True,
        )  # [S*B, L_pred+1, dim_z]
        z_path_mc = z_path_flat.view(n_trajs, batch_size, z_path_flat.shape[1], self.dim_z)  # [S, B, L_pred+1, dim_z]
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
        initial_state,
        delta_tps,
        n_trajs: int = 1,
    ):
        """
        Helper forecast from unbatched latent anchor

        Args:
          initial_state: [dim_z]
          delta_tps: [L_pred]
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (no state noise, observation mean only).
            - >1: stochastic rollout with both state and observation noise.

        Returns:
          x_samples:
            - [L_pred, dim_x] when n_trajs == 1
            - [S, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [L_pred, dim_z] when n_trajs == 1
            - [S, L_pred, dim_z] when n_trajs > 1
        """
        if n_trajs < 1:
            raise ValueError(f"Expected n_trajs >= 1, got {n_trajs}")
        
        ## normalize initial state and delta_tps so that they have a batch dimension
        initial_state = initial_state.unsqueeze(0)  # [1, dim_z]
        delta_tps = delta_tps.unsqueeze(0)  # [1, L_pred]
        
        x_samples, z_samples = self._forecast_from_state_batched(
            initial_state=initial_state,
            delta_tps=delta_tps,
            n_trajs=n_trajs,
        )
        if n_trajs == 1:
            x_samples = x_samples.squeeze(0) # [L_pred, dim_x]
            z_samples = z_samples.squeeze(0) # [L_pred, dim_z]
        else:
            # Unbatched API: remove singleton batch dimension.
            x_samples = x_samples.squeeze(1)  # [S, L_pred, dim_x]
            z_samples = z_samples.squeeze(1)  # [S, L_pred, dim_z]
        
        return x_samples, z_samples # [L_pred, dim_x], [L_pred, dim_z] when n_trajs == 1; [S, L_pred, dim_x], [S, L_pred, dim_z] when n_trajs > 1
    

    @torch.no_grad()
    def forecast_from_context(
        self,
        x,
        mask,
        time_steps,
        time_steps_to_predict,
        n_trajs: int = 1,
    ):
        """
        Forecast future trajectory on `time_steps_to_predict` using autoregressive one-step transitions.
        Args:
          x: [L_obs, dim_x]
          mask: [L_obs, dim_x], observation-availability mask.
          time_steps: [L_obs]
          time_steps_to_predict: [L_pred]
          n_trajs: number of rollout trajectories.
            - 1: deterministic rollout (no state noise, observation mean only).
            - >1: stochastic rollout with both state and observation noise.

        Returns:
          x_samples:
            - [L_pred, dim_x] when n_trajs == 1
            - [S, L_pred, dim_x] when n_trajs > 1
          z_samples:
            - [L_pred, dim_z] when n_trajs == 1
            - [S, L_pred, dim_z] when n_trajs > 1
        """
        # Stage 1: Infer the last latent state from the observed context
        x_batched = x.unsqueeze(0) # [1, L_obs, dim_x]
        time_steps_batched = time_steps.unsqueeze(0) # [1, L_obs]
        mask_batched = mask.unsqueeze(0)  # [1, L_obs, dim_x]
        mu_zq, _ = self.smooth(x_batched, mask_batched, time_steps_batched)
        initial_state = mu_zq[0, -1]  # [dim_z]
        t_init = time_steps[-1]  # scalar
        
        # Stage 2: Compute the time deltas for the prediction horizon
        time_steps_to_predict = torch.cat([t_init.unsqueeze(0), time_steps_to_predict], dim=0)  # [L_pred+1]
        delta_tps_to_predict = time_steps_to_predict[1:] - time_steps_to_predict[:-1]  # [L_pred]

        x_samples, z_samples = self.forecast_from_state(
            initial_state=initial_state,
            delta_tps=delta_tps_to_predict,
            n_trajs=n_trajs,
        ) # [S, L_pred, dim_x], [S, L_pred, dim_z]
        
        return x_samples, z_samples
