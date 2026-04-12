import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

EPS = 1.0e-12


class KL():
    """
    Gaussian KL utilities.

    This helper centralizes diagonal/full Gaussian KL computations used by
    ELBO, SDE exact-loss branches, and VAE-style regularization.
    """

    def __init__(self):
        pass

    @staticmethod
    def reduction(x, reduction = 'mean'):
        """
        Apply scalar reduction to a tensor.

        Args:
          x: tensor with any shape.
          reduction: None, "mean", or "sum".
        """
        if reduction is None:
            return x
        elif reduction == 'mean':
            return x.mean()
        elif reduction == 'sum':
            return x.sum()
        else:
            raise ValueError('unrecognized reduction method')

    @classmethod
    def normal(cls, mean_q, var_q, mean_p, var_p, reduction=None):
        """
        KL between diagonal-covariance Gaussians.

        Math (elementwise):
          q = N(mean_q, var_q), p = N(mean_p, var_p)
          returns KL(q || p), then applies reduction.

        Generic shape contract:
          mean_q, var_q: [...], parameters of q
          mean_p, var_p: [...], parameters of p
          all tensors must be broadcast-compatible.

        Reduction:
          - None: returns elementwise KL tensor with broadcasted shape
          - "mean": average over all elements
          - "sum": sum over all elements
        """
        q_dist = Normal(mean_q, var_q**0.5)
        p_dist = Normal(mean_p, var_p**0.5)
        kl = kl_divergence(q_dist, p_dist)
        return cls.reduction(kl, reduction = reduction)

    @classmethod
    def normal_full(
        cls,
        mean_q: torch.Tensor,
        cov_q: torch.Tensor,
        mean_p: torch.Tensor,
        cov_p: torch.Tensor,
        reduction: str | None = "mean",
        eps: float = EPS,
    ) -> torch.Tensor:
        """
        KL between full-covariance Gaussians.

        Math (per event):
          q = N(mean_q, cov_q), p = N(mean_p, cov_p)
          KL(q || p)
          = 0.5 * [ tr(cov_p^{-1} cov_q)
                    + (mean_p - mean_q)^T cov_p^{-1} (mean_p - mean_q)
                    + log|cov_p| - log|cov_q| - D ].

        Generic shape contract:
          mean_q, mean_p: [..., D]
          cov_q, cov_p: [..., D, D]
          leading dimensions must be broadcast-compatible.
        """
        if mean_q.shape[-1] != mean_p.shape[-1]:
            raise ValueError(
                f"Event-dimension mismatch: mean_q last dim {mean_q.shape[-1]} vs mean_p {mean_p.shape[-1]}."
            )
        if cov_q.shape[-1] != cov_q.shape[-2] or cov_p.shape[-1] != cov_p.shape[-2]:
            raise ValueError("Expected square trailing matrix dimensions for cov_q and cov_p.")
        if cov_q.shape[-1] != mean_q.shape[-1] or cov_p.shape[-1] != mean_p.shape[-1]:
            raise ValueError("Covariance event dimension must match mean event dimension.")

        d = mean_q.shape[-1]
        lead = torch.broadcast_shapes(mean_q.shape[:-1], mean_p.shape[:-1], cov_q.shape[:-2], cov_p.shape[:-2])
        mean_q = mean_q.expand(*lead, d)
        mean_p = mean_p.expand(*lead, d)
        cov_q = cov_q.expand(*lead, d, d)
        cov_p = cov_p.expand(*lead, d, d)

        n = 1
        for s in lead:
            n *= s
        mean_q_f = mean_q.reshape(n, d)
        mean_p_f = mean_p.reshape(n, d)
        cov_q_f = cov_q.reshape(n, d, d)
        cov_p_f = cov_p.reshape(n, d, d)

        eye = torch.eye(d, device=cov_p.device, dtype=cov_p.dtype).unsqueeze(0).repeat(n, 1, 1)
        cov_p_inv = torch.linalg.solve(cov_p_f + eps * eye, eye)

        trace = torch.diagonal(torch.bmm(cov_p_inv, cov_q_f), dim1=-2, dim2=-1).sum(dim=-1)
        mean_diff = (mean_p_f - mean_q_f).unsqueeze(-1)
        quad = torch.bmm(mean_diff.transpose(1, 2), torch.bmm(cov_p_inv, mean_diff)).squeeze(-1).squeeze(-1)
        logdet_q = torch.logdet(cov_q_f + eps * eye)
        logdet_p = torch.logdet(cov_p_f + eps * eye)

        kl = 0.5 * (trace + quad + logdet_p - logdet_q - d)
        return cls.reduction(kl, reduction=reduction)


class ELBO(nn.Module):
    """
    Generic ELBO components used across models.

    Reconstruction term:
      - MSE mode:   L_rec = ||x - x_hat||^2
      - NLL mode:   L_rec = -log N(x | x_hat, sigma_x^2)

    KL term for diagonal Gaussians:
      KL(q || p) where q = N(mu_q, var_q), p = N(mu_p, var_p).
    """

    def __init__(self, loss_type='MSE', reduction='mean'):
        super(ELBO, self).__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == 'MSE':
            self.loss_obj = nn.MSELoss(reduction=self.reduction)
        elif loss_type == 'NLL':
            self.loss_obj = nn.GaussianNLLLoss(reduction=self.reduction)
        else:
            raise ValueError(f'Unrecognized loss type: {loss_type}. Supported types are MSE and NLL.')


    def calc_reconstruction_err(self, targets, preds_mean, preds_var=None, target_mask=None):
        """
        Reconstruction term for the configured likelihood.

        - MSE mode: ||targets - preds_mean||^2
        - NLL mode: -log N(targets | preds_mean, preds_var)

        Args:
          targets: [..., dim_x]
          preds_mean: [..., dim_x]
          preds_var: optional [..., dim_x], required in `NLL` mode.
          target_mask: optional [..., dim_x]; masked entries are selected before reduction.

        Returns:
          Scalar tensor when `reduction in {"mean","sum"}`; otherwise tensor
          following the underlying PyTorch loss reduction behavior.
        """
        
        if target_mask is not None:
            assert torch.any(target_mask != 0), "target_mask must not be all zeros"
            target_mask = target_mask.bool()
            preds_mean = torch.masked_select(preds_mean, target_mask)
            targets = torch.masked_select(targets, target_mask)
            if preds_var is not None:
                preds_var = torch.masked_select(preds_var, target_mask)
        
        if self.loss_type == 'NLL':
            assert preds_var is not None
            return self.loss_obj(preds_mean, targets, preds_var)
        else:
            return self.loss_obj(preds_mean, targets)


    def calc_kl_divergence(self, mean_q, var_q, mean_p, var_p):
        """
        Diagonal Gaussian KL term:
          KL(N(mean_q, var_q) || N(mean_p, var_p))

        Args:
          mean_q, var_q, mean_p, var_p: tensors with matching shape `[..., D]`.
        """
        kl = KL.normal(
            mean_q=mean_q.reshape(-1, 1),
            var_q=var_q.reshape(-1, 1),
            mean_p=mean_p.reshape(-1, 1),
            var_p=var_p.reshape(-1, 1),
            reduction=self.reduction,
        )
        return kl

    
    def calc_entropy(self, mean, var):
        """
        Entropy of a diagonal Gaussian:
          H(N(mean, var))

        Args:
          mean, var: tensors with shape `[..., D]`.
        """

        p = Normal(mean, var**0.5)
        entropy = p.entropy()
        
        return KL.reduction(entropy, reduction=self.reduction)

