import torch

from ..configs.sde import SDEModelConfig, SDEOptConfig
from .base import LightningBase
from ...losses import SDEELBO
from ..modules import LatentSDEModel


class SDESSM(LightningBase):
    """
    Latent Neural SDE Lightning module with selectable ident-latent-sde loss mode.

    Pipeline:
      x -> posterior encoder -> sampled latent path z_{1:T}
      -> drift/diffusion nets produce (mu_hat, sigma_hat) on increments
      -> decoder reconstructs x from z.

    Objective:
      loss = recon_err + beta * kl_div

    where `kl_div` is chosen by opt_configs['kl_mode']:
      - 'mc':    MC increment KL surrogate (ident-latent-sde style)
      - 'exact': closed-form KL between increment Gaussians
      - 'vae':   standard VAE KL on posterior latents

    Reconstruction term:
      L_rec = -log N(x_shifted | x_hat_plus, x_sigma2 I)
    with fixed decoder variance `x_sigma2` (equivalent to scaled MSE up to a constant).

    Training transition pipeline (per batch):
      1) Posterior path:
         q_mu, q_sigma = Encoder(x, mask, t)
         z ~ q(z|x),  inc_t = z_t - z_{t-1}

      2) SDE increment stats:
         mu_hat_t = f_theta(t_{t-1}, z_in_{t-1})
         sigma_hat_t = g_theta(t_{t-1}, z_in_{t-1})
         where z_in is:
           - z (sampled posterior path) for mode='mc'
           - q_mu (posterior mean path) for mode in {'exact','vae'}

      3) Plus-one branch:
         z_{t+1}^+ = z_t + mu_hat_t * dt_t + diffusion_noise_t
         x_{t+1}^+ = Decoder(z_{t+1}^+)
         L_rec_plus = -log N(x_{2:T} | x^+_{2:T}, x_sigma2 I)

      4) Total objective:
         loss = recon_err + beta * kl_div

    Batch contract:
      (L denotes context/window length in a batch sample.)
      batch_data["x_traj"]: [B, L, dim_x]
      batch_data["x_mask"]: [B, L, dim_x]
      batch_data["x_tps"]:  [B, L] (or [L] at dataloader boundary, expanded in LightningBase)
      prediction requires:
      batch_data["y_tps"]:  [T_pred] or [B, T_pred]
    """
    def __init__(self, model_configs: SDEModelConfig, opt_configs: SDEOptConfig):
        super().__init__(model_configs, opt_configs)

        self.kl_mode = self.opt_configs.kl_mode
        if self.kl_mode == "vae" and self.model_configs.z_sigma_type != "diag":
            raise ValueError(
                "SDESSM requires model_configs.z_sigma_type='diag' when opt_configs.kl_mode='vae'."
            )
        self.elbo = SDEELBO(
            kl_mode=self.kl_mode,
            loss_type=self.opt_configs.loss_type,
            reduction=self.opt_configs.loss_reduction,
        )
        self.beta_max = self.opt_configs.beta_max
        self.warmup_epochs = self.opt_configs.warmup_epochs
        self.mc_samples = self.opt_configs.mc_samples

        self.model = LatentSDEModel(
            dim_x=self.model_configs.dim_obs,
            dim_z=self.model_configs.dim_latent,
            ## encoder
            enc_rnn_hidden_dim=self.model_configs.enc_rnn_hidden_dim,
            enc_rnn_num_layers=self.model_configs.enc_rnn_num_layers,
            enc_proj_dim=self.model_configs.enc_proj_dim,
            ## latent SDE transition
            drift_net_hidden_dim=self.model_configs.drift_net_hidden_dim,
            diffusion_net_hidden_dim=self.model_configs.diffusion_net_hidden_dim,
            z_sigma_type=self.model_configs.z_sigma_type,
            ## observation decoder
            dec_hidden_dim=self.model_configs.dec_hidden_dim,
            ## optional arguments: observation noise variance, encoder direction, dropout
            x_sigma2=self.model_configs.x_sigma2,
            enc_direction=self.model_configs.enc_direction,
            dropout=self.model_configs.dropout,
        )

    def _step(self, batch_data):
        """
        Run one SDE train/validation step and return reconstruction/KL/loss outputs.

        This function:
          1) calls the latent SDE model reconstruction path for training tensors,
          2) computes plus-one reconstruction and KL terms,
          3) combines them into the total objective.

        Returns:
          recon_err: scalar tensor
          kl_div: scalar tensor
          loss: scalar tensor

        Args:
          batch_data:
            x_traj: [B, T, dim_x]
            x_mask: [B, T, dim_x]
            x_tps:  [B, T]

        Notes:
          - uses plus-one reconstruction/latent regularization only.
        """
        x = batch_data["x_traj"]  # [B, L, dim_x]
        x_mask = batch_data["x_mask"]  # [B, L, dim_x]
        tps = self.get_tps_batched(batch_data, key="x_tps", batch_size=x.shape[0])  # [B, L]
        (
            x_plus_mean,
            x_plus_sigma2,
            mu_zq,
            sigma_zq,
            sampled_z,
        ) = self.model.reconstruct(
            x=x,
            mask=x_mask,
            time_steps=tps,
            mc_samples=self.mc_samples,
        )
        target = x[:, 1:, :]
        target_mask = x_mask[:, 1:, :]

        z_anchor = sampled_z[:, :-1, :] if self.kl_mode == "mc" else mu_zq[:, :-1, :]
        tps_in = tps[:, :-1]  # [B, L-1]
        dt = (tps[:, 1:] - tps[:, :-1]).unsqueeze(-1)  # [B, L-1, 1]
        mu_inc_p, sigma_inc_p = self.model.latent_transition.compute_increment_params(
            z_in=z_anchor,
            tps_in=tps_in,
        )

        recon_err = self.elbo.calc_reconstruction_err(
            targets=target,
            preds_mean=x_plus_mean,
            preds_var=x_plus_sigma2,
            target_mask=target_mask,
        )
        
        inc = sampled_z[:, 1:, :] - sampled_z[:, :-1, :]  # [B, L-1, Z]
        kl_div = self.elbo.calc_kl_divergence(
            q_mu=mu_zq,
            q_sigma=sigma_zq,
            mu_inc_p=mu_inc_p,
            sigma_inc_p=sigma_inc_p,
            inc=inc,
            dt=dt,
        )
        loss = recon_err + self._get_beta() * kl_div
        return recon_err, kl_div, loss
