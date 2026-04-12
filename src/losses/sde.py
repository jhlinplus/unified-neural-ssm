import torch
import torch.nn as nn

from .common import EPS, ELBO, KL

class MultivariateGaussianNLLLoss(nn.Module):
    """
    Multivariate Gaussian NLL for full-covariance outputs, analogous to
    `nn.GaussianNLLLoss` but with matrix covariance factors.

    Generic shape contract:
      input: [..., D], Gaussian mean
      target: [..., D], observations
      scale_tril: [..., D, D], lower-triangular covariance factor L

    Computes (up to additive constant):
      0.5 * ((x - mu)^T Sigma^{-1} (x - mu) + log|Sigma|)
    with Sigma = L L^T.

    Per-event math (for residual r = x - mu):

      nll(r; Sigma)
      = 0.5 * ( r^T Sigma^{-1} r + log det(Sigma) )

    and the module then applies reduction over all leading dimensions:
      - "none": returns [...], one NLL per event
      - "mean": average over all returned elements
      - "sum": sum over all returned elements
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction}.")
        self.reduction = reduction

    @classmethod
    def apply_reduction(cls, x: torch.Tensor, reduction: str):
        """Apply `"none"`, `"mean"`, or `"sum"` reduction to per-event NLL values."""
        if reduction == "none":
            return x
        if reduction == "mean":
            return x.mean()
        if reduction == "sum":
            return x.sum()
        raise ValueError(f"Unsupported reduction={reduction}.")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        scale_tril: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate multivariate Gaussian negative log-likelihood.

        Args:
          input: [..., D], mean tensor mu.
          target: [..., D], observation tensor x.
          scale_tril: [..., D, D], covariance factor L such that Sigma = L L^T.

        Returns:
          If reduction="none": tensor of shape [...] with per-event NLL values.
          If reduction in {"mean","sum"}: scalar tensor.
        """
        if input.shape != target.shape:
            raise ValueError(f"Expected input and target to share shape, got {input.shape} and {target.shape}.")
        if scale_tril.ndim < 2:
            raise ValueError(f"Expected scale_tril.ndim >= 2, got {scale_tril.ndim}.")
        if scale_tril.shape[-1] != scale_tril.shape[-2]:
            raise ValueError(f"Expected square trailing matrix in scale_tril, got {scale_tril.shape[-2:]}.")

        dim_z = scale_tril.shape[-1]
        if input.shape[-1] != dim_z:
            raise ValueError(
                f"Event-dimension mismatch: input/target last dim {input.shape[-1]} vs scale_tril dim {dim_z}."
            )

        try:
            torch.broadcast_shapes(input.shape[:-1], scale_tril.shape[:-2])
        except RuntimeError as e:
            raise ValueError(
                f"Leading dims are not broadcast-compatible: input {input.shape[:-1]} vs scale_tril {scale_tril.shape[:-2]}."
            ) from e

        cov = torch.matmul(scale_tril, scale_tril.transpose(-1, -2))

        eye = torch.eye(dim_z, device=cov.device, dtype=cov.dtype).view(*([1] * (cov.ndim - 2)), dim_z, dim_z)
        cov = cov + EPS * eye
        inv_cov = torch.linalg.inv(cov)

        resid = (target - input).unsqueeze(-1)
        quad = torch.matmul(resid.transpose(-1, -2), torch.matmul(inv_cov, resid)).squeeze(-1).squeeze(-1)
        logdet = torch.logdet(cov)
        nll = 0.5 * (quad + logdet)
        return self.apply_reduction(nll, reduction=self.reduction)


class SDEMcKL(nn.Module):
    """
    MC-style latent SDE KL surrogate from ident-latent-sde.

    Matches sampled increments against SDE-implied increment statistics.
    For diagonal sigma, follows the original expression in
    ident-latent-sde-main/losses.py::kl_div_cholesky.

    Conceptually:
      KL_MC ~ E[(Delta z - mu_hat dt)^T (Sigma2 dt)^{-1} (Delta z - mu_hat dt)]
              + log|Sigma2 dt|

    In this code, `sigma_hat` is interpreted as:
      - const: single scalar standard deviation
      - diag: standard deviations with shape [B, T-1, Z]
      - lower-triangular factor L when shape is [B, T-1, Z, Z]
    """

    def __init__(self):
        super().__init__()
        self.gaussian_nll = nn.GaussianNLLLoss(reduction="none")
        self.multivariate_gaussian_nll = MultivariateGaussianNLLLoss(reduction="mean")

    @staticmethod
    def _prepare_full_gaussian_terms(
        p_chol: torch.Tensor,
        dt: torch.Tensor,
        eps: float = EPS,
    ):
        """
        Precompute reusable linear-algebra terms for the full-covariance MC KL branch.

        Args:
          p_chol: [B, T, D, D], per-step lower-triangular factor L_t.
          dt: [1, T, 1] or [B, T, 1], per-step positive time increments.
          eps: minimum clamp value for `dt`.

        Returns:
          Tuple:
            bsz, n_steps, dim_z,
            dt_expanded [B,T,1],
            dt_flat [B*T,1,1],
            p_chol_flat [B*T,D,D],
            p_inv_flat [B*T,D,D],
            logdet_l [B*T].
        """
        if p_chol.ndim != 4:
            raise ValueError(f"Expected p_chol.ndim=4, got {p_chol.ndim}")

        bsz, n_steps, dim_z, _ = p_chol.shape
        dt = dt.clamp_min(eps)
        dt_expanded = dt.expand(bsz, n_steps, 1)
        dt_flat = dt_expanded.reshape(bsz * n_steps, 1, 1)

        p_chol_flat = p_chol.reshape(bsz * n_steps, dim_z, dim_z)
        eye = torch.eye(dim_z, device=p_chol.device, dtype=p_chol.dtype).unsqueeze(0).repeat(bsz * n_steps, 1, 1)
        p_inv_flat = torch.cholesky_solve(eye, p_chol_flat)
        logdet_l = p_chol_flat.det().log()

        return bsz, n_steps, dim_z, dt_expanded, dt_flat, p_chol_flat, p_inv_flat, logdet_l

    def _forward_diag(self, mu_hat: torch.Tensor, sigma_hat: torch.Tensor, inc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Diagonal MC surrogate from ident-latent-sde formulation.
        """
        # Use GaussianNLL as the base module and rescale:
        # NLL_diag = 0.5 * [ (inc-mu*dt)^2/(sigma^2*dt) + log(sigma^2*dt) ] (elementwise)
        # MC surrogate diagonal branch equals 2 * sum_D NLL_diag, averaged over (B,T).
        mean = mu_hat * dt
        sigma2_inc_p = (sigma_hat ** 2) * dt
        nll_elem = self.gaussian_nll(input=mean, target=inc, var=sigma2_inc_p)  # [B, T, Z]
        return 2.0 * nll_elem.sum(dim=-1).mean()

    def _forward_full(self, mu_hat: torch.Tensor, sigma_hat: torch.Tensor, inc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Full-covariance MC surrogate:
          E[(Delta z - mu dt)^T (Sigma2 dt)^{-1} (Delta z - mu dt)] + log|L| + log dt terms.
        """
        # Base multivariate NLL gives:
        #   NLL_full = 0.5 * quad + 0.5 * D * log(dt) + log|L|
        # Original MC surrogate full branch is:
        #   quad + D * log(dt) + log|L| = 2 * NLL_full - log|L|.
        mean = mu_hat * dt
        sqrt_dt = torch.sqrt(dt.clamp_min(EPS))
        while sqrt_dt.ndim < sigma_hat.ndim:
            sqrt_dt = sqrt_dt.unsqueeze(-1)
        scaled_tril = sigma_hat * sqrt_dt
        base_nll = self.multivariate_gaussian_nll(
            input=mean,
            target=inc,
            scale_tril=scaled_tril,
        )
        _, _, _, _, _, p_chol_flat, _, _ = self._prepare_full_gaussian_terms(
            p_chol=sigma_hat,
            dt=dt,
            eps=EPS,
        )
        logdet_l_mean = p_chol_flat.det().log().mean()
        return 2.0 * base_nll - logdet_l_mean

    def forward(self, mu_hat: torch.Tensor, sigma_hat: torch.Tensor, inc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Port of ident-latent-sde-main/losses.py::kl_div_cholesky adapted to batched sequences.

        Args:
            mu_hat: [B, T-1, Z]
            sigma_hat: scalar std (const), [B, T-1, Z] (diag), or [B, T-1, Z, Z] (full factor)
            inc: [B, T-1, Z]
            dt: [1, T-1, 1] or [B, T-1, 1]

        Returns:
            scalar loss tensor
        """
        if sigma_hat.ndim == 0:
            sigma_hat = sigma_hat * torch.ones_like(mu_hat)
        dt = dt.clamp_min(EPS)
        if sigma_hat.ndim == 3:
            return self._forward_diag(mu_hat=mu_hat, sigma_hat=sigma_hat, inc=inc, dt=dt)
        if sigma_hat.ndim == 4:
            return self._forward_full(mu_hat=mu_hat, sigma_hat=sigma_hat, inc=inc, dt=dt)
        raise ValueError(
            f"Unsupported sigma_hat ndim={sigma_hat.ndim}. "
            "Expected 0 (const scalar), 3 (diag), or 4 (full)."
        )


class SDEExactKL(nn.Module):
    """
    Exact Gaussian-KL branch from ident-latent-sde.

    Compares two increment Gaussians:
      q_inc = N(q_mu_inc, q_sigma2_inc)
      p_inc = N(mu_inc_p * dt, Sigma2_inc_p * dt), where
        - const/diag: Sigma2_inc_p = diag(sigma_inc_p^2) (const scalar is broadcast across latent dims)
        - full: Sigma2_inc_p = L L^T

    with closed-form KL(q_inc || p_inc), adapted to batched sequences.

    Shape conventions:
      - diagonal case:
          q_mu: [B, T-1, Z], q_sigma2_inc: [B, T-1, Z], sigma_inc_p: [B, T-1, Z]
      - full case:
          q_mu: [B, T-1, Z], q_sigma2_inc: [B, T-1, Z, Z], sigma_inc_p_chol: [B, T-1, Z, Z]
    """

    def __init__(self):
        super().__init__()

    def _forward_diag(
        self,
        q_mu_inc: torch.Tensor,
        q_sigma2_inc: torch.Tensor,
        mu_inc_p: torch.Tensor,
        sigma_inc_p: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diagonal exact branch:
          KL( N(q_mu_inc, q_sigma2_inc) || N(mu_inc_p * dt, sigma_inc_p^2 * dt) ).
        """
        model_inc_mean = mu_inc_p * dt
        sigma2_inc_p = (sigma_inc_p ** 2) * dt
        kl_elem = KL.normal(
            mean_q=q_mu_inc,
            var_q=q_sigma2_inc,
            mean_p=model_inc_mean,
            var_p=sigma2_inc_p,
            reduction=None,
        )
        return kl_elem.sum(dim=-1).mean()

    def _forward_full(
        self,
        q_mu_inc: torch.Tensor,
        q_sigma2_inc: torch.Tensor,
        mu_inc_p: torch.Tensor,
        sigma_inc_p_chol: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full exact branch:
          KL( N(q_mu_inc, q_sigma2_inc) || N(mu_inc_p * dt, L L^T * dt) ).
        """
        sqrt_dt = torch.sqrt(dt.clamp_min(EPS))
        while sqrt_dt.ndim < sigma_inc_p_chol.ndim:
            sqrt_dt = sqrt_dt.unsqueeze(-1)
        p_chol_scaled = sigma_inc_p_chol * sqrt_dt
        Sigma2_inc_p = torch.matmul(p_chol_scaled, p_chol_scaled.transpose(-1, -2))
        model_inc_mean = mu_inc_p * dt
        return KL.normal_full(
            mean_q=q_mu_inc,
            cov_q=q_sigma2_inc,
            mean_p=model_inc_mean,
            cov_p=Sigma2_inc_p,
            reduction="mean",
            eps=EPS,
        )

    def forward(
        self,
        q_mu_inc: torch.Tensor,
        q_sigma2_inc,
        mu_inc_p: torch.Tensor,
        sigma_inc_p_chol,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Port of ident-latent-sde-main/losses.py::kl_div_exact adapted to batched sequences.

        Args:
          q_mu_inc: [B, T-1, Z], posterior increment means.
          q_sigma2_inc: [B, T-1, Z] (diag variance) or [B, T-1, Z, Z] (full covariance).
          mu_inc_p: [B, T-1, Z], model drift output used for increment means.
          sigma_inc_p_chol: [B, T-1, Z] (diag scale) or [B, T-1, Z, Z] (Cholesky factor).
          dt: [1, T-1, 1] or [B, T-1, 1], used to scale model increment mean/covariance.

        Returns:
          scalar loss tensor
        """
        if sigma_inc_p_chol.ndim == 0:
            sigma_inc_p_chol = sigma_inc_p_chol * torch.ones_like(q_mu_inc)
        if q_sigma2_inc.ndim == 3 and sigma_inc_p_chol.ndim == 3:
            return self._forward_diag(
                q_mu_inc=q_mu_inc,
                q_sigma2_inc=q_sigma2_inc,
                mu_inc_p=mu_inc_p,
                sigma_inc_p=sigma_inc_p_chol,
                dt=dt,
            )
        if q_sigma2_inc.ndim == 4 and sigma_inc_p_chol.ndim == 4:
            return self._forward_full(
                q_mu_inc=q_mu_inc,
                q_sigma2_inc=q_sigma2_inc,
                mu_inc_p=mu_inc_p,
                sigma_inc_p_chol=sigma_inc_p_chol,
                dt=dt,
            )
        raise ValueError(
            f"Unsupported ndim combination for exact loss: "
            f"q_sigma2_inc.ndim={q_sigma2_inc.ndim}, sigma_inc_p_chol.ndim={sigma_inc_p_chol.ndim}. "
            "Expected (3,3) for diag or (4,4) for full."
        )


class SDEVaeKL(nn.Module):
    """
    Standard VAE latent KL term used by ident-latent-sde 'vae' mode.

    For q(z|x) = N(mu, sigma^2) and p(z)=N(0, I):
      KL(q || p) = -1/2 * sum(1 + log sigma^2 - mu^2 - sigma^2)

    This keeps ident-latent-sde's reduction style (sum over all elements).
    """

    def forward(self, q_mu: torch.Tensor, q_sigma: torch.Tensor) -> torch.Tensor:
        """
        Standard VAE latent KL:
          KL( N(q_mu, diag(q_sigma^2)) || N(0, I) )
        with sum reduction.
        """
        # Same objective as ident-latent-sde-main/nets.py (vae branch),
        # implemented via shared diagonal-Gaussian KL helper.
        return KL.normal(
            mean_q=q_mu,
            var_q=q_sigma ** 2,
            mean_p=torch.zeros_like(q_mu),
            var_p=torch.ones_like(q_sigma),
            reduction="sum",
        )


class SDEELBO(nn.Module):
    """
    SDE-specific objective wrapper used by `SDESSM`.

    This class combines:
      - reconstruction loss (`MSE` or fixed-variance Gaussian `NLL`) via `ELBO`
      - SDE KL branch selection (`mc`, `exact`, `vae`)

    The reconstruction branch is selected by `loss_type`, and the KL branch
    is selected by `kl_mode`, so `SDESSM` can call:
      - `calc_reconstruction_err(...)`
      - `calc_kl_divergence(...)`
    in the same style as discrete/ODE trainers using `ELBO`.
    """

    def __init__(self, kl_mode: str, loss_type: str = "NLL", reduction: str = "mean"):
        super().__init__()
        if kl_mode not in {"mc", "exact", "vae"}:
            raise ValueError(f"Unsupported kl_mode={kl_mode}. Expected one of {{'mc','exact','vae'}}.")
        if loss_type not in {"MSE", "NLL"}:
            raise ValueError(f"Unsupported loss_type={loss_type}. Expected one of {{'MSE','NLL'}}.")
        self.kl_mode = kl_mode
        self.loss_type = loss_type
        self.recon_loss = ELBO(loss_type=loss_type, reduction=reduction)
        if kl_mode == "mc":
            self.kl_loss = SDEMcKL()
        elif kl_mode == "exact":
            self.kl_loss = SDEExactKL()
        else:
            self.kl_loss = SDEVaeKL()

    @staticmethod
    def _triangle_vec_to_lower(tri_vec: torch.Tensor, dim_z: int) -> torch.Tensor:
        """
        Convert packed lower-triangular vectors to dense lower-triangular matrices.

        Args:
          tri_vec: [..., D*(D+1)//2]
          dim_z: D

        Returns:
          lower: [..., D, D]
        """
        lower = tri_vec.new_zeros(*tri_vec.shape[:-1], dim_z, dim_z)
        tril_idx = torch.tril_indices(dim_z, dim_z, device=tri_vec.device)
        lower[..., tril_idx[0], tril_idx[1]] = tri_vec
        return lower

    def _posterior_covariance_full(self, q_sigma: torch.Tensor, dim_z: int) -> torch.Tensor:
        """
        Build dense posterior covariance from SDE posterior parameterization.

        Args:
          q_sigma: [B, T, D] (diag) or [B, T, D*(D+1)//2] (packed Cholesky)
          dim_z: D

        Returns:
          cov: [B, T, D, D]
        """
        if q_sigma.shape[-1] == dim_z:
            return torch.diag_embed(q_sigma ** 2)
        lower = self._triangle_vec_to_lower(q_sigma, dim_z)
        return torch.matmul(lower, lower.transpose(-1, -2))

    def calc_reconstruction_err(
        self,
        targets: torch.Tensor,
        preds_mean: torch.Tensor,
        preds_var: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        SDE reconstruction error using the configured ELBO reconstruction branch.

        Args:
          targets: [B_like, T, X]
          preds_mean: [B_like, T, X]
          preds_var: [B_like, T, X] when `loss_type='NLL'`; ignored in `MSE` mode.
          target_mask: optional [B_like, T, X]

        Returns:
          Scalar reconstruction loss (for default `reduction="mean"`/`"sum"`).
        """
        if self.loss_type == "NLL" and preds_var is None:
            raise ValueError("preds_var is required when SDEELBO.loss_type='NLL'.")
        return self.recon_loss.calc_reconstruction_err(
            targets=targets,
            preds_mean=preds_mean,
            preds_var=preds_var,
            target_mask=target_mask,
        )

    def mc_kl(
        self,
        mu_inc_p: torch.Tensor,
        sigma_inc_p: torch.Tensor,
        inc: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        MC increment KL surrogate branch.

        Args:
          mu_inc_p: [B_like, L-1, D], model drift outputs on increment anchors.
          sigma_inc_p: [B_like, L-1, D] or [B_like, L-1, D, D], model diffusion factors/scales.
          inc: [B_like, L-1, D]
          dt: [1, L-1, 1]
        """
        return self.kl_loss(mu_hat=mu_inc_p, sigma_hat=sigma_inc_p, inc=inc, dt=dt)

    def exact_kl(
        self,
        q_mu: torch.Tensor,
        q_sigma: torch.Tensor,
        mu_inc_p: torch.Tensor,
        sigma_inc_p: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact KL branch with posterior increment Gaussian construction.

        Args:
          q_mu: [B_like, L, D]
          q_sigma: scalar posterior std (const), [B_like, L, D] (diag), or [B_like, L, D*(D+1)//2] (full).
          mu_inc_p: [B_like, L-1, D], model drift outputs on increment anchors.
          sigma_inc_p: scalar transition std (const), [B_like, L-1, D] (diag), or [B_like, L-1, D, D] (full factor).
          dt: [1, L-1, 1]
        """
        q_mu_inc = q_mu[:, 1:, :] - q_mu[:, :-1, :]
        dim_z = q_mu.shape[-1]
        if q_sigma.ndim == 0:
            q_sigma_diag = q_sigma * torch.ones_like(q_mu)
            posterior_is_diag = True
        else:
            q_sigma_diag = q_sigma
            posterior_is_diag = q_sigma.shape[-1] == dim_z
        diffusion_is_full = sigma_inc_p.ndim == 4

        if posterior_is_diag:
            q_sigma2_inc_diag = (q_sigma_diag[:, 1:, :] ** 2) + (q_sigma_diag[:, :-1, :] ** 2)
            q_sigma2_inc = torch.diag_embed(q_sigma2_inc_diag) if diffusion_is_full else q_sigma2_inc_diag
        else:
            q_sigma2_full = self._posterior_covariance_full(q_sigma_diag, dim_z=dim_z)
            q_sigma2_inc = q_sigma2_full[:, 1:, :, :] + q_sigma2_full[:, :-1, :, :]

        return self.kl_loss(
            q_mu_inc=q_mu_inc,
            q_sigma2_inc=q_sigma2_inc,
            mu_inc_p=mu_inc_p,
            sigma_inc_p_chol=sigma_inc_p,
            dt=dt,
        )

    def vae_kl(self, q_mu: torch.Tensor, q_sigma: torch.Tensor) -> torch.Tensor:
        """
        VAE posterior KL branch.

        Args:
          q_mu: [B_like, L, D]
          q_sigma: [B_like, L, D]
        """
        if q_sigma.ndim == 0:
            raise ValueError("SDEELBO.vae_kl expects per-dimension posterior std, got scalar const.")
        return self.kl_loss(q_mu=q_mu, q_sigma=q_sigma)

    def calc_kl_divergence(
        self,
        q_mu: torch.Tensor,
        q_sigma: torch.Tensor,
        mu_inc_p: torch.Tensor,
        sigma_inc_p: torch.Tensor,
        inc: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch to the configured SDE KL branch.

        Args:
          q_mu: [B_like, L, D]
          q_sigma: scalar posterior std (const), [B_like, L, D] (diag), or [B_like, L, D*(D+1)//2] (full).
          mu_inc_p: [B_like, L-1, D], model drift outputs on increment anchors.
          sigma_inc_p: scalar transition std (const), [B_like, L-1, D] (diag), or [B_like, L-1, D, D] (full factor).
          inc: [B_like, L-1, D]
          dt: [1, L-1, 1]

        Returns:
          Scalar KL loss for the configured `kl_mode`.
        """
        if self.kl_mode == "mc":
            return self.mc_kl(mu_inc_p=mu_inc_p, sigma_inc_p=sigma_inc_p, inc=inc, dt=dt)
        if self.kl_mode == "exact":
            return self.exact_kl(
                q_mu=q_mu,
                q_sigma=q_sigma,
                mu_inc_p=mu_inc_p,
                sigma_inc_p=sigma_inc_p,
                dt=dt,
            )
        return self.vae_kl(q_mu=q_mu, q_sigma=q_sigma)
