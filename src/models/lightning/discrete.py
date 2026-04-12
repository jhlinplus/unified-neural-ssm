import torch

from ..configs.discrete import DiscreteModelConfig, DiscreteOptConfig
from .base import LightningBase
from ...losses import ELBO
from ..modules import LatentDiscreteModel


class DiscreteSSM(LightningBase):
    """
    Discrete latent state-space Lightning module.

    Objective per batch (plus-one, shifted):
      L = L_rec + beta * KL

    Batch contract:
      batch_data["x_traj"]: [B, L, dim_x]
      batch_data["x_mask"]: [B, L, dim_x]
      batch_data["x_tps"]:  [L] or [B, L]
    """

    def __init__(self, model_configs: DiscreteModelConfig, opt_configs: DiscreteOptConfig):
        super().__init__(model_configs, opt_configs)

        self.elbo = ELBO(
            loss_type=self.opt_configs.loss_type,
            reduction=self.opt_configs.loss_reduction,
        )
        self.beta_max = self.opt_configs.beta_max
        self.warmup_epochs = self.opt_configs.warmup_epochs
        self.mc_samples = self.opt_configs.mc_samples

        self.model = LatentDiscreteModel(
            dim_x=self.model_configs.dim_obs,
            dim_z=self.model_configs.dim_latent,
            ## encoder
            enc_proj_dim=self.model_configs.enc_proj_dim,
            enc_rnn_hidden_dim=self.model_configs.enc_rnn_hidden_dim,
            enc_rnn_num_layers=self.model_configs.enc_rnn_num_layers,
            ## latent state-space transition
            ar_proj_hidden_dim=self.model_configs.ar_proj_hidden_dim,
            ar_mean_net_hidden_dim=self.model_configs.ar_mean_net_hidden_dim,
            ar_sigma_net_hidden_dim=self.model_configs.ar_sigma_net_hidden_dim,
            z_sigma_type=self.model_configs.z_sigma_type,
            ## observation decoder
            dec_hidden_dim=self.model_configs.dec_hidden_dim,
            ## optional arguments: encoder mask processor, observation noise variance, dropout
            mask_processor_kernel_size=self.model_configs.mask_processor_kernel_size,
            x_sigma2=self.model_configs.x_sigma2,
            dropout=self.model_configs.dropout,
        )

    def _step(self, batch_data: dict):
        """
        Args:
            batch_data: dict with `x_traj`, `x_mask`, `x_tps`.

        Returns:
            recon_err: scalar tensor, reconstruction error.
            kl_div: scalar tensor, KL divergence.
            loss: scalar tensor, total loss (recon_err + beta * kl_div).
        """
        x = batch_data["x_traj"]
        x_mask = batch_data["x_mask"]
        tps = self.get_tps_batched(batch_data, key="x_tps", batch_size=x.shape[0])
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
            mc_samples=self.mc_samples,
        )
        target = x[:, 1:, :]
        target_mask = x_mask[:, 1:, :]
        preds_var = x_plus_sigma2 if self.elbo.loss_type == "NLL" else None
        recon_err = self.elbo.calc_reconstruction_err(
            targets=target,
            preds_mean=x_plus_mean,
            preds_var=preds_var,
            target_mask=target_mask,
        )
        kl_div = self.elbo.calc_kl_divergence(
            mean_q=mu_zq[:, 1:, :],
            var_q=sigma2_zq[:, 1:, :],
            mean_p=mu_zp_plus,
            var_p=sigma2_zp_plus,
        )
        loss = recon_err + self._get_beta() * kl_div
        return recon_err, kl_div, loss
