from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    n_epochs: int
    batch_size: int
    lr: float
    scheduler_type: str
    weight_decay: float = 1.0e-6
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 0.0
    limit_val_batches: int = 100
    gradient_accumulation_steps: int = 1
    early_stopping: int | None = None

    factor: float | None = None
    patience: int | None = None
    step_size: int | None = None
    gamma: float = 0.5
    milestones: list[int] | None = None

    def __post_init__(self):
        if self.scheduler_type not in {"ReduceLROnPlateau", "StepLR", "MultiStepLR"}:
            raise ValueError(
                f"Unsupported scheduler_type={self.scheduler_type}. "
                "Expected one of {'ReduceLROnPlateau', 'StepLR', 'MultiStepLR'}."
            )
        if self.scheduler_type == "ReduceLROnPlateau":
            if self.factor is None or self.patience is None:
                raise ValueError("ReduceLROnPlateau requires `factor` and `patience`.")
        elif self.scheduler_type == "StepLR":
            if self.step_size is None:
                raise ValueError("StepLR requires `step_size`.")
        elif self.scheduler_type == "MultiStepLR":
            if self.milestones is None:
                raise ValueError("MultiStepLR requires `milestones`.")
