from abc import ABC, abstractmethod
from dataclasses import is_dataclass

import torch
import lightning as L


class LightningBase(ABC, L.LightningModule):
    """
    Shared Lightning scaffold for SSM model trainers.

    Notation:
      T_obs: full trajectory length before partitioning/windowing.
      L: context/window length in each batch sample.

    Expected batch convention across subclasses:
      x_traj: [B, L, dim_x]
      x_mask: [B, L, dim_x]
      x_tps:  [L] or [B, L]
      optional prediction keys:
      y_traj: [B, T_pred, dim_x]
      y_mask: [B, T_pred, dim_x]
      y_tps:  [T_pred] or [B, T_pred]
    """
    def __init__(self, model_configs, opt_configs):
        
        super(LightningBase, self).__init__()
        
        self.save_hyperparameters()
        self.model_configs = model_configs
        self.opt_configs = opt_configs
        
        print('=================================')
        print(self.model_configs)
        print('----------------')
        print(self.opt_configs)
        print('=================================')

    @property
    def model_configs(self):
        return self._model_configs

    @model_configs.setter
    def model_configs(self, model_configs):
        if not is_dataclass(model_configs):
            raise TypeError("model_configs must be a dataclass instance.")
        self._model_configs = model_configs

    @property
    def opt_configs(self):
        return self._opt_configs

    @opt_configs.setter
    def opt_configs(self, opt_configs):
        if not is_dataclass(opt_configs):
            raise TypeError("opt_configs must be a dataclass instance.")
        self._opt_configs = opt_configs


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.opt_configs.lr,
                                     weight_decay=self.opt_configs.weight_decay)

        if self.opt_configs.scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                            mode='min',
                            factor=self.opt_configs.factor,
                            patience=self.opt_configs.patience)
            lr_scheduler = {'scheduler':scheduler, 'monitor':'val_loss'}
        elif self.opt_configs.scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                            step_size=self.opt_configs.step_size,
                            gamma=self.opt_configs.gamma)
            lr_scheduler = {'scheduler':scheduler, 'interval': 'epoch'}
        elif self.opt_configs.scheduler_type == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=self.opt_configs.milestones,
                            gamma=self.opt_configs.gamma)
            lr_scheduler = {'scheduler':scheduler, 'interval': 'epoch'}
        else:
            raise ValueError('unrecognized scheduler_type')

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def log_epoch_metrics(self, values, prefix):
        log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log_kwargs["batch_size"] = self.opt_configs.batch_size
        self.log_dict({f"{prefix}_{k}": v for k, v in values.items()}, **log_kwargs)

    @staticmethod
    def get_tps_1d(batch_data: dict, key: str) -> torch.Tensor:
        """
        Return a 1D timeline tensor `[T]` from batched timestamp input.

        Accepts:
          - already 1D `[T]`
          - batched `[B, T]` with identical rows.
        """
        tps = batch_data[key]
        if tps.ndim == 1:
            return tps
        if tps.ndim == 2:
            if not torch.allclose(tps, tps[:1, :].expand_as(tps)):
                raise ValueError(f"{key} must be identical across the batch")
            return tps[0]
        raise ValueError(f"Unsupported shape for {key}: {tps.shape}")

    @staticmethod
    def get_tps_batched(batch_data: dict, key: str, batch_size: int) -> torch.Tensor:
        """
        Return a batched timeline tensor `[B, T]` from timestamp input.

        Accepts:
          - already batched `[B, T]`
          - 1D `[T]`, which is repeated across batch dimension.
        """
        tps = batch_data[key]
        if tps.ndim == 2:
            return tps
        if tps.ndim == 1:
            return tps.unsqueeze(0).repeat(batch_size, 1)
        raise ValueError(f"Unsupported shape for {key}: {tps.shape}")

    @staticmethod
    def unsqueeze_all(*inputs, dim=0):
        return tuple(x.unsqueeze(dim) for x in inputs)

    @staticmethod
    def squeeze_all(*inputs, dim=0):
        return tuple(x.squeeze(dim) for x in inputs)

    @staticmethod
    def linear_warmup_beta(current_epoch: int, beta_max: float, warmup_epochs: int | None) -> float:
        if warmup_epochs is None:
            return beta_max
        return beta_max * min(warmup_epochs, current_epoch + 1) / warmup_epochs

    def _get_beta(self) -> float:
        """
        Get beta coefficient for KL term using linear warmup schedule.
        
        Subclasses can override this for custom warmup schedules.
        """
        return self.linear_warmup_beta(
            current_epoch=self.current_epoch,
            beta_max=self.beta_max,
            warmup_epochs=self.warmup_epochs,
        )

    @abstractmethod
    def _step(self, batch_data):
        """
        Subclass training/validation computation.

        Args:
            batch_data: batch dict following the class batch contract.

        Returns:
            tuple whose first three entries are:
              recon_err: scalar tensor
              kl_div: scalar tensor
              loss: scalar tensor
        """
        raise NotImplementedError("_step not implemented")

    def training_step(self, batch_data, batch_idx):
        recon_err, kl_div, loss, *_ = self._step(batch_data)
        self.log_epoch_metrics({"err": recon_err, "kl": kl_div, "loss": loss}, prefix="train")
        return loss


    def validation_step(self, batch_data, batch_idx):
        recon_err, kl_div, loss, *_ = self._step(batch_data)
        self.log_epoch_metrics({"err": recon_err, "kl": kl_div, "loss": loss}, prefix="val")
