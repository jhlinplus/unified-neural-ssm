import torch

from ...losses import ODELoss
from ..base import LightningBase
from .config import ODEModelConfig, ODEOptConfig
from .network import LatentODEModel


class ODESSM(LightningBase):
    """
    Latent Neural ODE Lightning module.

    Pipeline:
      x -> encoder -> z0 ~ q(z0|x) -> ODE solve -> x_hat

    Objective:
      Configurable via `opt_configs.loss_mode`:
      - "elbo": mean_s(-log p(x_target | z^(s))) + beta * KL(q(z0|x) || N(0, I))
      - "iwae": -logsumexp_s(log p(x_target | z^(s)) - beta * KL(q(z0|x) || N(0, I)))

    Batch contract:
      (L denotes context/window length in a batch sample.)
      batch_data["x_traj"]: [B, L, dim_x]
      batch_data["x_mask"]: [B, L, dim_x]
      batch_data["x_tps"]:  [L] or [B, L]
      
      optionally for prediction mode:
      batch_data["y_traj"]: [B, T_pred, dim_x]
      batch_data["y_mask"]: [B, T_pred, dim_x]
      batch_data["y_tps"]:  [T_pred] or [B, T_pred]
    """
    def __init__(self, model_configs: ODEModelConfig, opt_configs: ODEOptConfig):
        super().__init__(model_configs, opt_configs)

        self.ode_loss = ODELoss(
            loss_type=self.opt_configs.loss_type,
            reduction=self.opt_configs.loss_reduction,
        )
        self.train_strategy = self.opt_configs.train_strategy
        self.loss_mode = self.opt_configs.loss_mode
        self.mc_samples = self.opt_configs.mc_samples
        self.beta_max = self.opt_configs.beta_max
        self.warmup_epochs = self.opt_configs.warmup_epochs

        self.model = LatentODEModel(
            dim_x=self.model_configs.dim_obs,
            dim_z=self.model_configs.dim_latent,
            ## encoder
            enc_rnn_hidden_dim=self.model_configs.enc_rnn_hidden_dim,
            enc_rnn_num_layers=self.model_configs.enc_rnn_num_layers,
            enc_proj_dim=self.model_configs.enc_proj_dim,
            ## latent ODE transition
            ode_net_hidden_dim=self.model_configs.ode_net_hidden_dim,
            ## observation decoder
            dec_hidden_dim=self.model_configs.dec_hidden_dim,
            ## optional arguments: observation noise variance, encoder direction, dropout
            x_sigma2=self.model_configs.x_sigma2,
            enc_direction=self.model_configs.enc_direction,
            dropout=self.model_configs.dropout,
        )

    def _step(self, batch_data):
        """
        Args:
            batch_data: batch dictionary following the class batch contract.

        Returns:
            recon_err: scalar tensor
            kl_div: scalar tensor
            loss: scalar tensor
        """
        x = batch_data["x_traj"]      # [B, L_ctx, dim_x]
        x_mask = batch_data["x_mask"] # [B, L_ctx, dim_x]
        x_tps = self.get_tps_batched(batch_data, "x_tps", batch_size=x.shape[0])  # [B, L_ctx]

        if self.train_strategy == "reconstruct":
            target = x
            target_mask = x_mask
            tps_target = x_tps
        else:
            target = batch_data["y_traj"]       # [B, T_target, dim_x]
            target_mask = batch_data["y_mask"]  # [B, T_target, dim_x]
            tps_target = self.get_tps_batched(batch_data, "y_tps", batch_size=x.shape[0])  # [B, T_target]

        x_mean, x_sigma2, z0_mean, z0_std, _ = self.model.reconstruct(
            x=x,
            mask=x_mask,
            time_steps=x_tps,
            time_steps_to_reconstruct=tps_target,
            mc_samples=self.mc_samples,
        )

        log_lik = self.ode_loss.calc_reconstruction_loglik(
            targets=target,
            preds_mean=x_mean,
            preds_var=x_sigma2,
            target_mask=target_mask,
        )  # [S]
        kl_div = self.ode_loss.calc_kl_divergence(
            z0_mean=z0_mean,
            z0_std=z0_std,
        )  # scalar

        recon_err = -log_lik.mean()  # scalar
        beta_kl = self._get_beta()
        if self.loss_mode == "elbo":
            loss = self.ode_loss.calc_elbo_loss(
                log_likelihood=log_lik,
                kl_div=kl_div,
                beta_kl=beta_kl,
            )  # scalar
        elif self.loss_mode == "iwae":
            loss = self.ode_loss.calc_iwae_loss(
                log_likelihood=log_lik,
                kl_div=kl_div,
                beta_kl=beta_kl,
            )  # scalar
        else:
            raise ValueError(
                f"Unsupported loss_mode={self.loss_mode}. Expected one of {{'elbo', 'iwae'}}."
            )
        return recon_err, kl_div, loss
