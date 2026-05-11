import torch

from ...losses import ELBO
from ..base import LightningBase
from .config import NowcastModelConfig, NowcastOptConfig
from .network import LatentNowcastModel


class NowcastSSM(LightningBase):
    """
    Mask-aware discrete latent state-space Lightning module.

    Objective per batch (same-step aligned):
      L = L_rec + beta * KL

    Batch contract:
      batch_data["x_traj"]: [B, L, dim_x]
      batch_data["x_mask"]: [B, L, dim_x]
      batch_data["x_tps"]:  [L] or [B, L]
      optional decoder guidance mask:
      batch_data["x_dec_mask"]: [B, L-1, dim_x]
    """

    def __init__(self, model_configs: NowcastModelConfig, opt_configs: NowcastOptConfig):
        super().__init__(model_configs, opt_configs)

        self.elbo = ELBO(
            loss_type=self.opt_configs.loss_type,
            reduction=self.opt_configs.loss_reduction,
        )
        self.beta_max = self.opt_configs.beta_max
        self.warmup_epochs = self.opt_configs.warmup_epochs
        self.mc_samples = self.opt_configs.mc_samples

        self.model = LatentNowcastModel(
            dim_x=self.model_configs.dim_obs,
            dim_z=self.model_configs.dim_latent,
            enc_proj_dim=self.model_configs.enc_proj_dim,
            enc_rnn_hidden_dim=self.model_configs.enc_rnn_hidden_dim,
            enc_rnn_num_layers=self.model_configs.enc_rnn_num_layers,
            ar_proj_hidden_dim=self.model_configs.ar_proj_hidden_dim,
            ar_mean_net_hidden_dim=self.model_configs.ar_mean_net_hidden_dim,
            ar_sigma_net_hidden_dim=self.model_configs.ar_sigma_net_hidden_dim,
            z_sigma_type=self.model_configs.z_sigma_type,
            dec_hidden_dim=self.model_configs.dec_hidden_dim,
            mask_processor_kernel_size=self.model_configs.mask_processor_kernel_size,
            x_sigma2=self.model_configs.x_sigma2,
            dropout=self.model_configs.dropout,
        )

    def _get_unguided_sample_ratio(self) -> float:
        return self.linear_warmup_beta(
            current_epoch=self.current_epoch,
            beta_max=self.opt_configs.unguided_sample_ratio,
            warmup_epochs=self.warmup_epochs,
        )

    def _sample_guidance_mask(self, observed_target_mask: torch.Tensor) -> torch.Tensor:
        batch_size = observed_target_mask.shape[0]
        guidance_mask = torch.zeros_like(observed_target_mask)

        unguided_sample_ratio = self._get_unguided_sample_ratio()
        n_unguided = round(batch_size * unguided_sample_ratio)
        n_unguided = max(0, min(batch_size, n_unguided))
        unguided_samples = torch.zeros(batch_size, dtype=torch.bool, device=observed_target_mask.device)
        if n_unguided > 0:
            unguided_indices = torch.randperm(batch_size, device=observed_target_mask.device)[:n_unguided]
            unguided_samples[unguided_indices] = True

        guided_samples = ~unguided_samples
        prob = self.opt_configs.pseudo_guidance_mask_prob
        if prob is not None and torch.any(guided_samples):
            if isinstance(prob, tuple):
                low, high = prob
                probs = low + (high - low) * torch.rand(batch_size, device=observed_target_mask.device)
            else:
                probs = torch.full(
                    (batch_size,),
                    fill_value=prob,
                    device=observed_target_mask.device,
                )
            probs = probs.view(batch_size, 1, 1)
            sampled_guidance = (torch.rand_like(observed_target_mask) < probs).to(observed_target_mask.dtype)
            sampled_guidance = sampled_guidance * observed_target_mask
            guidance_mask[guided_samples] = sampled_guidance[guided_samples]

        # When there is observed target mass, keep at least one observed entry for reconstruction
        # so the default training regime does not degenerate to KL-only updates.
        flat_guidance = guidance_mask.view(batch_size, -1)
        flat_observed = observed_target_mask.view(batch_size, -1)
        for sample_idx in range(batch_size):
            if unguided_samples[sample_idx]:
                continue
            if torch.any(flat_observed[sample_idx] != 0) and torch.all(
                flat_guidance[sample_idx] == flat_observed[sample_idx]
            ):
                first_obs_idx = torch.nonzero(flat_observed[sample_idx] != 0, as_tuple=False)[0, 0]
                flat_guidance[sample_idx, first_obs_idx] = 0.0
        return flat_guidance.view_as(observed_target_mask)

    @staticmethod
    def _calc_guidance_sample_fractions(guidance_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_unguided = torch.all(guidance_mask == 0, dim=(1, 2))
        unguided_frac = is_unguided.to(guidance_mask.dtype).mean()
        guided_frac = guidance_mask.new_tensor(1.0) - unguided_frac
        return unguided_frac, guided_frac

    @staticmethod
    def _validate_guidance_subset(
        guidance_mask: torch.Tensor,
        observed_target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if guidance_mask.shape != observed_target_mask.shape:
            raise ValueError(
                f"guidance_mask.shape={guidance_mask.shape}; "
                f"observed_target_mask.shape={observed_target_mask.shape}; expecting the same shape"
            )
        guidance_mask = guidance_mask.to(observed_target_mask.dtype)
        if torch.any(guidance_mask > observed_target_mask):
            raise ValueError("x_dec_mask must be a subset of the observed target mask x_mask on aligned steps.")
        return guidance_mask

    def _calc_recon_err(
        self,
        targets: torch.Tensor,
        preds_mean: torch.Tensor,
        preds_var: torch.Tensor | None,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(target_mask != 0):
            if self.opt_configs.allow_all_zero_target_mask:
                return preds_mean.new_tensor(0.0)
            raise ValueError(
                "Aligned reconstruction target mask is all zeros. "
                "Provide a less aggressive x_dec_mask or set allow_all_zero_target_mask=True."
            )
        return self.elbo.calc_reconstruction_err(
            targets=targets,
            preds_mean=preds_mean,
            preds_var=preds_var,
            target_mask=target_mask,
        )

    def _step(self, batch_data: dict, sample_if_missing: bool = True):
        x = batch_data["x_traj"]
        x_mask = batch_data["x_mask"]
        tps = self.get_tps_batched(batch_data, key="x_tps", batch_size=x.shape[0])

        target = x[:, 1:, :]
        observed_target_mask = x_mask[:, 1:, :]

        provided_guidance_mask = batch_data.get("x_dec_mask")
        if provided_guidance_mask is not None:
            guidance_mask = self._validate_guidance_subset(
                guidance_mask=provided_guidance_mask,
                observed_target_mask=observed_target_mask,
            )
        elif sample_if_missing:
            guidance_mask = self._sample_guidance_mask(observed_target_mask)
        else:
            guidance_mask = torch.zeros_like(observed_target_mask)
        unguided_frac, guided_frac = self._calc_guidance_sample_fractions(guidance_mask)

        guidance_x = target * guidance_mask
        # Only non-guided observed entries contribute reconstruction loss; the
        # guided entries are consumed as same-step correction inputs instead.
        target_mask = observed_target_mask * (1 - guidance_mask)

        (
            x_plus_mean,
            x_plus_sigma2,
            mu_zq,
            sigma2_zq,
            mu_zp_plus,
            sigma2_zp_plus,
        ) = self.model.reconstruct(
            x=x,
            mask=x_mask,
            time_steps=tps,
            guidance_x_plus=guidance_x if torch.any(guidance_mask != 0) else None,
            guidance_mask_plus=guidance_mask if torch.any(guidance_mask != 0) else None,
            mc_samples=self.mc_samples,
        )

        preds_var = x_plus_sigma2 if self.elbo.loss_type == "NLL" else None
        recon_err = self._calc_recon_err(
            targets=target,
            preds_mean=x_plus_mean,
            preds_var=preds_var,
            target_mask=target_mask,
        )
        # KL remains between the posterior path and the unguided latent prior;
        # the guidance correction affects reconstruction but not this prior term.
        kl_div = self.elbo.calc_kl_divergence(
            mean_q=mu_zq[:, 1:, :],
            var_q=sigma2_zq[:, 1:, :],
            mean_p=mu_zp_plus,
            var_p=sigma2_zp_plus,
        )
        loss = recon_err + self._get_beta() * kl_div
        return recon_err, kl_div, loss, unguided_frac, guided_frac

    def training_step(self, batch_data, batch_idx):
        recon_err, kl_div, loss, unguided_frac, guided_frac = self._step(batch_data)
        self.log_epoch_metrics(
            {
                "err": recon_err,
                "kl": kl_div,
                "loss": loss,
                "unguided_frac": unguided_frac,
                "guided_frac": guided_frac,
            },
            prefix="train",
        )
        return loss

    def validation_step(self, batch_data, batch_idx):
        recon_err, kl_div, loss, unguided_frac, guided_frac = self._step(batch_data, sample_if_missing=False)
        self.log_epoch_metrics(
            {
                "err": recon_err,
                "kl": kl_div,
                "loss": loss,
                "unguided_frac": unguided_frac,
                "guided_frac": guided_frac,
            },
            prefix="val",
        )
