import os
import math
import time

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import torch


class Printer(Callback):
    
    def __init__(self, verbosity=10):
        self.verbosity = verbosity
    
    def on_train_epoch_start(self, trainer, pl_module):
        ## time a epoch
        self.start_time = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        ## time a epoch then print
        elapsed_time = time.monotonic() - self.start_time
        if pl_module.current_epoch == 0 or (pl_module.current_epoch+1)%self.verbosity == 0:
            elogs = trainer.logged_metrics
            lrs = pl_module.optimizers().param_groups
            loss_str, err_str, kl_str = '', '', ''
            for key in ['train_loss', 'val_loss']:
                if key in elogs:
                    loss_str += f'{key}={elogs[key]:.4f}; '
            for key in ['train_err', 'val_err']:
                if key in elogs:
                    err_str += f'{key}={elogs[key]:.2f}; '
            for key in ['train_kl', 'val_kl']:
                if key in elogs:
                    kl_str += f'{key}={elogs[key]:.2f}; '
            print(f"EPOCH={pl_module.current_epoch+1:03d} (timer={elapsed_time:.0f}s/epoch); {loss_str}lr={float(lrs[0]['lr']):.1E}")
            if len(err_str) or len(kl_str):
                print(f'    - {err_str}{kl_str}')

            
def configure_trainer(opt_configs, gpu_id=None, output_dir='output/debug', verbosity=10,
                    detect_anomaly=False, fast_dev_run=False):
    
    ckpt_dir = os.path.join(output_dir, 'pl_ckpts')
    for ckpt_suffix in ['best','last']:
        ckpt_path = os.path.join(ckpt_dir, f'model-{ckpt_suffix}.ckpt')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f'{ckpt_path} removed')
    time.sleep(3)
    
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=f'model-best', monitor='val_loss', save_last=True)
    ckpt_callback.CHECKPOINT_NAME_LAST=f'model-last'

    printer = Printer(verbosity=verbosity)
    callbacks = [printer, ckpt_callback]
    if opt_configs.early_stopping is not None:
        early_stopper = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=opt_configs.early_stopping,
                                    verbose=False,
                                    mode="min")
        callbacks.append(early_stopper)
        
    use_gpu = gpu_id is not None and torch.cuda.is_available()
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = [gpu_id] if use_gpu else 1

    trainer = L.Trainer(accelerator=accelerator,
                        devices=devices,
                        max_epochs=opt_configs.n_epochs,
                        callbacks=callbacks,
                        enable_progress_bar=False,
                        check_val_every_n_epoch=opt_configs.check_val_every_n_epoch,
                        logger=TensorBoardLogger(save_dir=output_dir, name='tensorboard'),
                        gradient_clip_val=opt_configs.gradient_clip_val,
                        num_sanity_val_steps=0,
                        limit_val_batches=opt_configs.limit_val_batches,
                        accumulate_grad_batches=opt_configs.gradient_accumulation_steps,
                        precision=32,
                        detect_anomaly=detect_anomaly,
                        fast_dev_run=fast_dev_run)
    
    return trainer
