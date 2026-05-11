import torch
import torch.nn as nn

from ..common.nn import BiRNNEncoder, GaussianObsDecoder, MLP, STD_MIN_SCALE, Sampler


class NowcastTransition(nn.Module):
    """
    Mask-aware discrete latent prior plus guidance-correction block.

    Parameterizes the autonomous latent prior:
      p(z_t | z_{t-1}, dt_t)

    Guidance is consumed afterward as a same-step innovation correction on the
    predicted latent state, rather than being fused directly into the prior.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        proj_dim: int,
        mask_processor_kernel_size: int,
        mean_net_hidden_dim: int | list[int] | None = None,
        sigma_net_hidden_dim: int | list[int] | None = None,
        sigma_type: str = "const",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.proj_dim = proj_dim
        if sigma_type not in {"const", "diag"}:
            raise ValueError(
                f"Unsupported sigma_type={sigma_type}. Expected one of {{'const','diag'}}."
            )
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

        self.guidance_projection = MLP(
            input_dim=dim_x,
            output_dim=proj_dim,
            hidden_dim=[proj_dim],
            dropout=dropout,
            batch_norm=False,
        )

        self.mask_1dcnn = nn.Conv1d(
            dim_x, 2 * proj_dim, mask_processor_kernel_size, padding="same")
        self.mask_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * proj_dim, proj_dim),
        )
        self.guidance_gate = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.Sigmoid(),
        )
        self.correction_net = MLP(
            input_dim=2 * proj_dim,
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

    def _validate_guidance(
        self,
        guidance_x: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
        expected_lead_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if guidance_x is None:
            if guidance_mask is not None:
                raise ValueError(
                    "guidance_mask cannot be provided when guidance_x is None.")
            return None, None

        if guidance_x.shape[:-1] != expected_lead_shape or guidance_x.shape[-1] != self.dim_x:
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; expecting {expected_lead_shape + (self.dim_x,)}"
            )

        if guidance_mask is None:
            guidance_mask = torch.ones_like(guidance_x)
        elif guidance_mask.shape != guidance_x.shape:
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; guidance_mask.shape={guidance_mask.shape}; "
                "expecting the same shape"
            )
        return guidance_x, guidance_mask

    def compute_guidance_hidden(
        self,
        guidance_x: torch.Tensor,
        guidance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if guidance_x.ndim != 3:
            raise ValueError(
                f"guidance_x.ndim={guidance_x.ndim}; expecting a batched sequence [B, T, dim_x]"
            )
        guidance_x, guidance_mask = self._validate_guidance(
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
            expected_lead_shape=tuple(guidance_x.shape[:-1]),
        )
        guidance_hidden = self.guidance_projection(guidance_x)
        mask_hidden = self.mask_projection(
            self.mask_1dcnn(guidance_mask.permute(0, 2, 1)).permute(0, 2, 1)
        )
        guidance_hidden = guidance_hidden + mask_hidden

        has_step_guidance = torch.any(guidance_mask != 0, dim=-1, keepdim=True)
        guidance_hidden = guidance_hidden * \
            has_step_guidance.to(guidance_hidden.dtype)
        return guidance_hidden

    def _prior_step_params(
        self,
        z_inputs: torch.Tensor,
        dt_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Private prior parameterization kernel on flattened inputs.

        Args:
          z_inputs: [N, dim_z]
          dt_inputs: [N, 1]

        Returns:
          z_next_mean: [N, dim_z]
          z_next_sigma2: [N, dim_z]
        """
        z_projected = self.z_projection(z_inputs)
        z_projected_with_t = torch.cat([z_projected, dt_inputs], dim=-1)
        z_next_mean = self.f_mean(z_projected_with_t)
        if self.sigma_type == "const":
            z_next_std_scalar = self.softplus(
                self.sigma_const_raw).to(z_next_mean) + STD_MIN_SCALE
            z_next_std = z_next_std_scalar.view(1, 1).expand(
                z_next_mean.shape[0], self.dim_z)
        else:
            z_next_std = self.softplus(self.f_sigma(
                z_projected_with_t)) + STD_MIN_SCALE
        z_next_sigma2 = z_next_std**2
        return z_next_mean, z_next_sigma2

    def _apply_guidance_correction(
        self,
        z_prior_inputs: torch.Tensor,
        guidance_hidden_inputs: torch.Tensor,
    ) -> torch.Tensor:
        z_projected = self.z_projection(z_prior_inputs)
        guidance_gate = self.guidance_gate(
            torch.cat([z_projected, guidance_hidden_inputs], dim=-1))
        gated_guidance = guidance_gate * guidance_hidden_inputs
        # Predict correction in latent space after the gate decides how much of
        # the innovation-derived guidance features should affect each channel.
        delta_z = self.correction_net(
            torch.cat([z_projected, gated_guidance], dim=-1))
        return z_prior_inputs + delta_z

    def one_step_transition(
        self,
        z_prev: torch.Tensor,
        delta_t: torch.Tensor,
        include_state_noise: bool = True,
    ) -> torch.Tensor:
        z_curr, z_curr_sigma2 = self._prior_step_params(
            z_inputs=z_prev, dt_inputs=delta_t)
        if include_state_noise:
            z_curr = Sampler.sample_normal(z_curr, z_curr_sigma2, mc_samples=1)
        return z_curr

    def compute_next_step_params(
        self,
        z_in: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute one-step latent prior parameters from pre-aligned inputs.

        Args:
          z_in: [B_like, T, dim_z]
          delta_t: [B_like, T, 1]

        Returns:
          z_next_mean: [B_like, T, dim_z]
          z_next_sigma2: [B_like, T, dim_z]
        """
        flat_z_prev = z_in.reshape(-1, self.dim_z)
        flat_dt = delta_t.reshape(-1, 1)
        z_next_mean_flat, z_next_sigma2_flat = self._prior_step_params(
            z_inputs=flat_z_prev, dt_inputs=flat_dt)
        z_next_mean = z_next_mean_flat.view_as(z_in)
        z_next_sigma2 = z_next_sigma2_flat.view_as(z_in)
        return z_next_mean, z_next_sigma2

    def apply_guidance_correction(
        self,
        z_prior: torch.Tensor,
        x_prior_mean: torch.Tensor,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply a same-step innovation correction to latent prior states.

        Args:
          z_prior: [B_like, T, dim_z] or [B_like, dim_z]
          x_prior_mean: same leading shape as z_prior with trailing dim_x
          guidance_x: optional observation guidance aligned with x_prior_mean
          guidance_mask: optional guidance-entry mask aligned with guidance_x

        Returns:
          z_corrected: same shape as z_prior
        """
        if guidance_x is None:
            if guidance_mask is not None:
                raise ValueError(
                    "guidance_mask cannot be provided when guidance_x is None.")
            return z_prior

        is_sequence = z_prior.ndim == 3
        if z_prior.ndim not in {2, 3}:
            raise ValueError(
                f"z_prior.ndim={z_prior.ndim}; expecting 2 or 3 dimensions with trailing dim_z={self.dim_z}."
            )
        expected_x_shape = z_prior.shape[:-1] + (self.dim_x,)
        if x_prior_mean.shape != expected_x_shape:
            raise ValueError(
                f"x_prior_mean.shape={x_prior_mean.shape}; expecting {expected_x_shape} to match z_prior."
            )
        if guidance_x.shape != expected_x_shape:
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; expecting {expected_x_shape} to match x_prior_mean."
            )
        if guidance_mask is None:
            guidance_mask = torch.ones_like(guidance_x)
        elif guidance_mask.shape != guidance_x.shape:
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; guidance_mask.shape={guidance_mask.shape}; "
                "expecting the same shape"
            )

        if not is_sequence:
            z_prior = z_prior.unsqueeze(1)
            x_prior_mean = x_prior_mean.unsqueeze(1)
            guidance_x = guidance_x.unsqueeze(1)
            guidance_mask = guidance_mask.unsqueeze(1)

        # Kalman-style innovation: observed guidance minus the model's own
        # observation prediction under the latent prior.
        innovation = (guidance_x - x_prior_mean) * guidance_mask
        guidance_hidden = self.compute_guidance_hidden(
            innovation.reshape(-1, 1, self.dim_x),
            guidance_mask.reshape(-1, 1, self.dim_x),
        ).squeeze(1)
        flat_z_prior = z_prior.reshape(-1, self.dim_z)
        z_corrected = self._apply_guidance_correction(
            flat_z_prior, guidance_hidden).view_as(z_prior)

        has_step_guidance = torch.any(guidance_mask != 0, dim=-1, keepdim=True)
        z_corrected = torch.where(has_step_guidance, z_corrected, z_prior)
        if not is_sequence:
            z_corrected = z_corrected.squeeze(1)
        return z_corrected

    def forward(
        self,
        z_prev: torch.Tensor,
        delta_t: torch.Tensor,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
        include_state_noise: bool = True,
        obs_decoder: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute one single-step predict-correct latent nowcast step.

        Args:
          z_prev: [B, dim_z]
          delta_t: [B, 1]
          guidance_x: optional [B, dim_x], same-step guidance for the predicted observation.
          guidance_mask: optional [B, dim_x], observed-entry mask for guidance_x.
          include_state_noise: whether to sample transition noise in the latent prior.
          obs_decoder: required when guidance_x is provided so the prior prediction can be decoded
            and corrected using innovation.

        Returns:
          z_corrected: [B, dim_z], latent state after optional same-step correction.
          mu_zprior: [B, dim_z], latent prior mean before correction.
          sigma2_zprior: [B, dim_z], latent prior variance before correction.
        """
        # The unguided latent prior remains the KL object even when the caller
        # requests a same-step correction afterward.
        mu_zprior, sigma2_zprior = self._prior_step_params(
            z_inputs=z_prev,
            dt_inputs=delta_t,
        )
        if include_state_noise:
            z_prior = Sampler.sample_normal(mu_zprior, sigma2_zprior, mc_samples=1)
        else:
            z_prior = mu_zprior
        if guidance_x is None:
            if guidance_mask is not None:
                raise ValueError(
                    "guidance_mask cannot be provided when guidance_x is None.")
            return z_prior, mu_zprior, sigma2_zprior
        if obs_decoder is None:
            raise ValueError(
                "obs_decoder is required when guidance_x is provided so forward(...) can apply "
                "the same-step guidance correction."
            )
        x_prior_mean, _ = obs_decoder(z_prior)
        z_corrected = self.apply_guidance_correction(
            z_prior=z_prior,
            x_prior_mean=x_prior_mean,
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
        )
        return z_corrected, mu_zprior, sigma2_zprior

    def forward_batched_sequence(
        self,
        z_prev: torch.Tensor,
        delta_t: torch.Tensor,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
        include_state_noise: bool = True,
        obs_decoder: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequence-aware wrapper around forward(...) for aligned [B_like, T, ...] inputs.

        Returns:
          z_corrected: [B_like, T, dim_z]
          mu_zprior: [B_like, T, dim_z]
          sigma2_zprior: [B_like, T, dim_z]
        """
        # Keep the aligned training path sequence-native to avoid flattening the
        # step axis only to expand it again inside the correction branch.
        mu_zprior, sigma2_zprior = self.compute_next_step_params(z_prev, delta_t)
        if include_state_noise:
            z_prior = Sampler.sample_normal(mu_zprior, sigma2_zprior, mc_samples=1)
        else:
            z_prior = mu_zprior
        if guidance_x is None:
            if guidance_mask is not None:
                raise ValueError("guidance_mask cannot be provided when guidance_x is None.")
            return z_prior, mu_zprior, sigma2_zprior
        if obs_decoder is None:
            raise ValueError(
                "obs_decoder is required when guidance_x is provided so forward_batched_sequence(...) "
                "can apply the same-step guidance correction."
            )
        x_prior_mean, _ = obs_decoder(z_prior)
        z_corrected = self.apply_guidance_correction(
            z_prior=z_prior,
            x_prior_mean=x_prior_mean,
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
        )
        return z_corrected, mu_zprior, sigma2_zprior


class LatentNowcastModel(nn.Module):
    """
    Mask-aware discrete latent state-space model for conditional nowcasting.

    End-to-end training path:
      x -> q(z_{1:L}|x) -> sample z -> p(z_t|z_{t-1}) -> correct(z_t, x_t^obs, m_t) -> p(x_t|z_t)

    Shape convention:
      x: [B, L, dim_x]
      time_steps: [B, L]
      latent states: [B, L, dim_z]

    Training follows the same latent-anchor regime as `LatentDiscreteModel`:
    one-step transitions are trained against the smoothed posterior path rather
    than against long free-running rollout states.
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
        self.latent_transition = NowcastTransition(
            dim_x=dim_x,
            dim_z=dim_z,
            proj_dim=ar_proj_hidden_dim,
            mask_processor_kernel_size=mask_processor_kernel_size,
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        guidance_x_plus: torch.Tensor | None = None,
        guidance_mask_plus: torch.Tensor | None = None,
        mc_samples: int = 1,
    ):
        return self.reconstruct(
            x=x,
            mask=mask,
            time_steps=time_steps,
            guidance_x_plus=guidance_x_plus,
            guidance_mask_plus=guidance_mask_plus,
            mc_samples=mc_samples,
        )

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        guidance_x_plus: torch.Tensor | None = None,
        guidance_mask_plus: torch.Tensor | None = None,
        mc_samples: int = 1,
    ):
        """
        Build aligned same-step reconstruction tensors for training.

        Args:
          x: [B, L, dim_x]
          mask: [B, L, dim_x], observation-availability mask for encoder.
          time_steps: [B, L]
          guidance_x_plus: optional [B, L-1, dim_x], observed guidance values for x_{2:T}.
          guidance_mask_plus: optional [B, L-1, dim_x], guidance mask for x_{2:T}.
          mc_samples: number of Monte Carlo posterior samples before MC averaging.

        Returns:
          x_plus_mean: [B, L-1, dim_x], decoded means for x_{2:T}.
          x_plus_sigma2: [B, L-1, dim_x], decoded variances for x_{2:T}.
          mu_zq: [B, L, dim_z], posterior latent means.
          sigma2_zq: [B, L, dim_z], posterior latent variances.
          mu_zp_plus: [B, L-1, dim_z], one-step latent prior means.
          sigma2_zp_plus: [B, L-1, dim_z], one-step latent prior variances.
        """
        mu_zq, sigma2_zq, _, _ = self.posterior_encoder(x, mask, time_steps)
        delta_t = (time_steps[:, 1:] - time_steps[:, :-1]).unsqueeze(-1)

        if mc_samples < 1:
            transition_anchor = mu_zq
        else:
            transition_anchor = Sampler.sample_normal(
                mu_zq,
                sigma2_zq,
                mc_samples=mc_samples,
                take_mc_average=True,
            )

        z_in = transition_anchor[:, :-1, :]
        # Use the same single-step predict-correct path as forecast. The returned
        # prior statistics remain the KL target, while reconstruction decodes the
        # corrected latent state.
        z_plus, mu_zp_plus, sigma2_zp_plus = self.latent_transition.forward_batched_sequence(
            z_prev=z_in,
            delta_t=delta_t,
            guidance_x=guidance_x_plus,
            guidance_mask=guidance_mask_plus,
            include_state_noise=False,
            obs_decoder=self.obs_decoder,
        )
        x_plus_mean, x_plus_sigma2 = self.obs_decoder(z_plus)
        return x_plus_mean, x_plus_sigma2, mu_zq, sigma2_zq, mu_zp_plus, sigma2_zp_plus

    @torch.no_grad()
    def smooth(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_zq, sigma2_zq, _, _ = self.posterior_encoder(x, mask, time_steps)
        return mu_zq, sigma2_zq

    @staticmethod
    def _normalize_guidance_batched(
        guidance_x: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        dim_x: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if guidance_x is None:
            if guidance_mask is not None:
                raise ValueError(
                    "guidance_mask cannot be provided when guidance_x is None.")
            return None, None
        if guidance_x.shape != (batch_size, seq_len, dim_x):
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; expecting {(batch_size, seq_len, dim_x)}"
            )
        if guidance_mask is None:
            guidance_mask = torch.ones_like(guidance_x)
        elif guidance_mask.shape != guidance_x.shape:
            raise ValueError(
                f"guidance_x.shape={guidance_x.shape}; guidance_mask.shape={guidance_mask.shape}; "
                "expecting the same shape"
            )
        return guidance_x, guidance_mask

    @torch.no_grad()
    def _forecast_from_state_batched(
        self,
        initial_state: torch.Tensor,
        delta_tps: torch.Tensor,
        n_trajs: int = 1,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
    ):
        """
        Helper forecast from normalized batched latent anchor.

        Args:
          initial_state: [B, dim_z]
          delta_tps: [B, L_pred]
          n_trajs: number of rollout trajectories.
          guidance_x: optional [B, L_pred, dim_x], observed values for each forecast step.
          guidance_mask: optional [B, L_pred, dim_x], observed-entry mask for each forecast step.

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
        seq_len = delta_tps.shape[1]
        guidance_x, guidance_mask = self._normalize_guidance_batched(
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            dim_x=self.dim_x,
        )

        if n_trajs == 1:
            z_prev = initial_state
            z_samples = []
            x_samples = []
            for step_idx in range(seq_len):
                # forward(...) performs the full single-step predict-correct
                # nowcast update before the latent is propagated onward.
                z_curr, _, _ = self.latent_transition.forward(
                    z_prev=z_prev,
                    delta_t=delta_tps[:, step_idx].unsqueeze(-1),
                    guidance_x=None if guidance_x is None else guidance_x[:, step_idx, :],
                    guidance_mask=None if guidance_mask is None else guidance_mask[:, step_idx, :],
                    include_state_noise=False,
                    obs_decoder=self.obs_decoder,
                )
                x_curr, _ = self.obs_decoder(z_curr)
                z_samples.append(z_curr)
                x_samples.append(x_curr)
                z_prev = z_curr
            return torch.stack(x_samples, dim=1), torch.stack(z_samples, dim=1)

        z_prev = initial_state.unsqueeze(0).expand(n_trajs, batch_size, self.dim_z).reshape(
            n_trajs * batch_size, self.dim_z
        )
        if guidance_x is not None:
            guidance_x = (
                guidance_x.unsqueeze(0)
                .expand(n_trajs, batch_size, seq_len, self.dim_x)
                .reshape(n_trajs * batch_size, seq_len, self.dim_x)
            )
            guidance_mask = (
                guidance_mask.unsqueeze(0)
                .expand(n_trajs, batch_size, seq_len, self.dim_x)
                .reshape(n_trajs * batch_size, seq_len, self.dim_x)
            )
        delta_tps = (
            delta_tps.unsqueeze(0)
            .expand(n_trajs, batch_size, seq_len)
            .reshape(n_trajs * batch_size, seq_len)
        )

        z_samples = []
        x_samples = []
        for step_idx in range(seq_len):
            # Stochastic forecast keeps transition noise in the prior, then
            # applies the same-step correction before decoding/saving the step.
            z_curr, _, _ = self.latent_transition.forward(
                z_prev=z_prev,
                delta_t=delta_tps[:, step_idx].unsqueeze(-1),
                guidance_x=None if guidance_x is None else guidance_x[:, step_idx, :],
                guidance_mask=None if guidance_mask is None else guidance_mask[:, step_idx, :],
                include_state_noise=True,
                obs_decoder=self.obs_decoder,
            )
            x_mean, x_sigma2 = self.obs_decoder(z_curr)
            x_curr = Sampler.sample_normal(
                mu=x_mean,
                var=x_sigma2,
                mc_samples=1,
                take_mc_average=False,
            ).squeeze(0)
            z_samples.append(z_curr)
            x_samples.append(x_curr)
            z_prev = z_curr

        z_samples = torch.stack(z_samples, dim=1).view(
            n_trajs, batch_size, seq_len, self.dim_z)
        x_samples = torch.stack(x_samples, dim=1).view(
            n_trajs, batch_size, seq_len, self.dim_x)
        return x_samples, z_samples

    @torch.no_grad()
    def forecast_from_state(
        self,
        initial_state: torch.Tensor,
        delta_tps: torch.Tensor,
        n_trajs: int = 1,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
    ):
        """
        Forecast from an unbatched latent anchor.

        Args:
          initial_state: [dim_z]
          delta_tps: [L_pred]
          n_trajs: number of rollout trajectories.
          guidance_x: optional [L_pred, dim_x]
          guidance_mask: optional [L_pred, dim_x]
        """
        if n_trajs < 1:
            raise ValueError(f"Expected n_trajs >= 1, got {n_trajs}")

        initial_state = initial_state.unsqueeze(0)
        delta_tps = delta_tps.unsqueeze(0)
        if guidance_x is not None:
            if guidance_x.shape != (delta_tps.shape[1], self.dim_x):
                raise ValueError(
                    f"guidance_x.shape={guidance_x.shape}; expecting {(delta_tps.shape[1], self.dim_x)}"
                )
            guidance_x = guidance_x.unsqueeze(0)
            if guidance_mask is not None:
                if guidance_mask.shape != guidance_x.squeeze(0).shape:
                    raise ValueError(
                        f"guidance_x.shape={guidance_x.squeeze(0).shape}; guidance_mask.shape={guidance_mask.shape}; "
                        "expecting the same shape"
                    )
                guidance_mask = guidance_mask.unsqueeze(0)

        x_samples, z_samples = self._forecast_from_state_batched(
            initial_state=initial_state,
            delta_tps=delta_tps,
            n_trajs=n_trajs,
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
        )
        if n_trajs == 1:
            x_samples = x_samples.squeeze(0)
            z_samples = z_samples.squeeze(0)
        else:
            x_samples = x_samples.squeeze(1)
            z_samples = z_samples.squeeze(1)
        return x_samples, z_samples

    @torch.no_grad()
    def forecast_from_context(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_steps: torch.Tensor,
        time_steps_to_predict: torch.Tensor,
        n_trajs: int = 1,
        guidance_x: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
    ):
        """
        Forecast future trajectory with optional same-step guidance assimilation.

        Args:
          x: [L_obs, dim_x]
          mask: [L_obs, dim_x]
          time_steps: [L_obs]
          time_steps_to_predict: [L_pred]
          guidance_x: optional [L_pred, dim_x], observed components of future x_t.
          guidance_mask: optional [L_pred, dim_x], observed-entry mask for guidance_x.
          n_trajs: number of rollout trajectories.
        """
        x_batched = x.unsqueeze(0)
        time_steps_batched = time_steps.unsqueeze(0)
        mask_batched = mask.unsqueeze(0)
        mu_zq, _ = self.smooth(x_batched, mask_batched, time_steps_batched)
        initial_state = mu_zq[0, -1]
        t_init = time_steps[-1]

        time_steps_to_predict = torch.cat(
            [t_init.unsqueeze(0), time_steps_to_predict], dim=0)
        delta_tps_to_predict = time_steps_to_predict[1:] - \
            time_steps_to_predict[:-1]

        return self.forecast_from_state(
            initial_state=initial_state,
            delta_tps=delta_tps_to_predict,
            n_trajs=n_trajs,
            guidance_x=guidance_x,
            guidance_mask=guidance_mask,
        )
