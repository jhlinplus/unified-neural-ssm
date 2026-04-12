import torch
import torch.nn as nn

from .common import KL

class ODELoss(nn.Module):
    """
    ODE-specific variational objective components.

    For S posterior samples:
      log w_s = log p_theta(x | z^{(s)}) - beta_KL * KL(q(z0|x) || p(z0)).

    Supports two aggregation modes:
      - ELBO-style: mean_s[-log p_theta(x|z^(s))] + beta_KL * KL
      - IWAE-style: -logsumexp_s(log w_s)

    In IWAE mode, the `-log S` normalization is omitted (constant w.r.t. parameters).
    """

    def __init__(self, loss_type: str = "NLL", reduction: str = "mean"):
        super().__init__()
        if loss_type not in {"MSE", "NLL"}:
            raise ValueError(
                f"Unsupported loss_type={loss_type}. Expected one of {{'MSE','NLL'}}."
            )
        if reduction not in {"mean", "sum"}:
            raise ValueError(
                f"Unsupported reduction={reduction}. Expected one of {{'mean','sum'}}."
            )
        self.loss_type = loss_type
        self.reduction = reduction
        self.gaussian_nll = nn.GaussianNLLLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def _reduce_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce per-sample flattened losses.

        Args:
          x: [S, N], flattened loss elements for each sample.

        Returns:
          [S], reduced loss per sample.
        """
        if self.reduction == "mean":
            return x.mean(dim=-1)
        return x.sum(dim=-1)

    def calc_reconstruction_loglik(
        self,
        targets: torch.Tensor,
        preds_mean: torch.Tensor,
        preds_var: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction log-likelihood surrogates.

        Args:
          targets: [B, T, X]
          preds_mean: [S, B, T, X] or [B, T, X]
          preds_var: [S, B, T, X] or [B, T, X], required when loss_type='NLL'
          target_mask: optional [B, T, X]

        Returns:
          log_likelihood: [S], per-sample values used in IWAE-style aggregation.
        """
        if preds_mean.ndim == 3:
            preds_mean = preds_mean.unsqueeze(0)  # [1, B, T, X]
        if preds_mean.ndim != 4:
            raise ValueError(f"Expected preds_mean with ndim 4, got {preds_mean.ndim}.")

        n_samples = preds_mean.shape[0]
        targets_expand = targets.unsqueeze(0).expand(n_samples, *targets.shape)

        if self.loss_type == "NLL":
            if preds_var is None:
                raise ValueError("preds_var is required when loss_type='NLL'.")
            if preds_var.ndim == 3:
                preds_var = preds_var.unsqueeze(0)  # [1, B, T, X]
            if preds_var.shape[0] == 1 and n_samples > 1:
                preds_var = preds_var.expand(n_samples, *preds_var.shape[1:])
            loss_elem = self.gaussian_nll(
                input=preds_mean,
                target=targets_expand,
                var=preds_var,
            )  # [S, B, T, X]
        else:
            loss_elem = self.mse(preds_mean, targets_expand)  # [S, B, T, X]

        if target_mask is not None:
            if not torch.any(target_mask != 0):
                raise ValueError("target_mask must contain at least one valid entry.")
            mask_expand = target_mask.unsqueeze(0).expand_as(loss_elem).bool()
            loss_flat = torch.masked_select(loss_elem, mask_expand).view(n_samples, -1)  # [S, N_valid]
        else:
            loss_flat = loss_elem.reshape(n_samples, -1)  # [S, N]

        nll_per_sample = self._reduce_per_sample(loss_flat)  # [S]
        return -nll_per_sample

    def calc_kl_divergence(self, z0_mean: torch.Tensor, z0_std: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(q(z0|x) || p(z0)) with p(z0)=N(0,I).

        Args:
          z0_mean: [B, Z]
          z0_std: [B, Z]

        Returns:
          scalar KL value under configured reduction.
        """
        return KL.normal(
            mean_q=z0_mean,
            var_q=z0_std ** 2,
            mean_p=torch.zeros_like(z0_mean),
            var_p=torch.ones_like(z0_std),
            reduction=self.reduction,
        )

    def calc_iwae_loss(
        self,
        log_likelihood: torch.Tensor,
        kl_div: torch.Tensor,
        beta_kl: float,
    ) -> torch.Tensor:
        """
        Compute IWAE-style aggregated loss.

        Args:
          log_likelihood: [S], per-sample reconstruction log-likelihood terms.
          kl_div: scalar KL value.
          beta_kl: KL coefficient.

        Returns:
          scalar loss = -logsumexp(log_likelihood - beta_kl * kl_div).
        """
        if log_likelihood.ndim != 1:
            raise ValueError(f"Expected log_likelihood shape [S], got {tuple(log_likelihood.shape)}")
        return -torch.logsumexp(log_likelihood - beta_kl * kl_div, dim=0)

    def calc_elbo_loss(
        self,
        log_likelihood: torch.Tensor,
        kl_div: torch.Tensor,
        beta_kl: float,
    ) -> torch.Tensor:
        """
        Compute ELBO-style loss from sample-wise log-likelihood terms.

        Args:
          log_likelihood: [S], per-sample reconstruction log-likelihood terms.
          kl_div: scalar KL value.
          beta_kl: KL coefficient.

        Returns:
          scalar loss = mean_s(-log_likelihood_s) + beta_kl * kl_div.
        """
        if log_likelihood.ndim != 1:
            raise ValueError(f"Expected log_likelihood shape [S], got {tuple(log_likelihood.shape)}")
        recon_err = -log_likelihood.mean()
        return recon_err + beta_kl * kl_div

