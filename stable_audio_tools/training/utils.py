from pytorch_lightning.loggers import WandbLogger, CometLogger
from ..interface.aeiou import pca_point_cloud

import wandb
import torch
import os
import math

def get_rank():
    """Get rank of current process."""

    print(os.environ.keys())

    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

class InverseLRHalfMix(torch.optim.lr_scheduler._LRScheduler):
    """Inverse decay with exponential warmup and a 50/50 residual base-LR mix.

    Closed form:
        base_lr * (1 - warmup ** (step + 1)) *
        (0.5 * (1 + step / inv_gamma) ** (-power) + 0.5)

    This decays like InverseLR early on, but asymptotes to 0.5 * base_lr
    instead of decaying toward zero.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if inv_gamma <= 0:
            raise ValueError('Invalid value for inv_gamma')
        if power < 0:
            raise ValueError('Invalid value for power')
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        inverse_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        mixed_mult = 0.5 * inverse_mult + 0.5
        return [base_lr * warmup * mixed_mult for base_lr in self.base_lrs]

class ExponentialWarmupPlateauLR(torch.optim.lr_scheduler._LRScheduler):
    """Exponential warmup to the optimizer base LR, then a flat plateau."""

    def __init__(self, optimizer, warmup=0., warmup_steps=0, last_epoch=-1):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        if warmup_steps < 0:
            raise ValueError('Invalid value for warmup_steps')
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _warmup_mult(self):
        if self.warmup == 0:
            return 1.

        step = max(self.last_epoch, 0)
        if step >= self.warmup_steps:
            return 1.

        warmup = 1 - self.warmup ** (step + 1)
        target = 1 - self.warmup ** (self.warmup_steps + 1)
        return min(1., warmup / target)

    def _get_closed_form_lr(self):
        warmup_mult = self._warmup_mult()
        return [base_lr * warmup_mult for base_lr in self.base_lrs]

class WarmupHoldCosineFloorLR(torch.optim.lr_scheduler._LRScheduler):
    """Exponential warmup, hold at base LR, cosine decay to a floor, then plateau."""

    def __init__(
            self,
            optimizer,
            warmup=0.,
            warmup_steps=0,
            hold_until_step=0,
            decay_until_step=1,
            final_lr=0.,
            last_epoch=-1
    ):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        if warmup_steps < 0:
            raise ValueError('Invalid value for warmup_steps')
        if hold_until_step < warmup_steps:
            raise ValueError('hold_until_step must be >= warmup_steps')
        if decay_until_step <= hold_until_step:
            raise ValueError('decay_until_step must be > hold_until_step')
        if final_lr < 0:
            raise ValueError('Invalid value for final_lr')
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.hold_until_step = hold_until_step
        self.decay_until_step = decay_until_step
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _warmup_mult(self):
        if self.warmup == 0:
            return 1.

        step = max(self.last_epoch, 0)
        warmup = 1 - self.warmup ** (step + 1)
        target = 1 - self.warmup ** (self.warmup_steps + 1)
        return min(1., warmup / target)

    def _get_closed_form_lr(self):
        step = max(self.last_epoch, 0)

        if step <= self.warmup_steps:
            return [base_lr * self._warmup_mult() for base_lr in self.base_lrs]

        if step <= self.hold_until_step:
            return list(self.base_lrs)

        if step >= self.decay_until_step:
            return [self.final_lr for _ in self.base_lrs]

        decay_position = (step - self.hold_until_step) / (self.decay_until_step - self.hold_until_step)
        cosine_mult = 0.5 * (1 + math.cos(math.pi * decay_position))
        return [
            self.final_lr + (base_lr - self.final_lr) * cosine_mult
            for base_lr in self.base_lrs
        ]

class PiecewiseLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Absolute LR schedule with linear or cosine interpolation between milestones."""

    def __init__(self, optimizer, milestones, last_epoch=-1):
        if not milestones:
            raise ValueError("milestones must contain at least one milestone")

        parsed_milestones = []
        for milestone in milestones:
            step = int(milestone["step"])
            lr = float(milestone["lr"])
            interpolation = milestone.get("interpolation", "linear")
            if step < 0:
                raise ValueError("milestone step must be >= 0")
            if lr < 0:
                raise ValueError("milestone lr must be >= 0")
            if interpolation not in {"linear", "cosine"}:
                raise ValueError("interpolation must be 'linear' or 'cosine'")
            parsed_milestones.append({
                "step": step,
                "lr": lr,
                "interpolation": interpolation,
            })

        parsed_milestones = sorted(parsed_milestones, key=lambda item: item["step"])
        for prev, curr in zip(parsed_milestones, parsed_milestones[1:]):
            if curr["step"] <= prev["step"]:
                raise ValueError("milestone steps must be strictly increasing")

        self.milestones = parsed_milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        lr = self._get_closed_form_lr_value()
        return [lr for _ in self.base_lrs]

    def _get_closed_form_lr_value(self):
        step = max(self.last_epoch, 0)

        if step <= self.milestones[0]["step"]:
            return self.milestones[0]["lr"]

        for prev, curr in zip(self.milestones, self.milestones[1:]):
            if step <= curr["step"]:
                span = curr["step"] - prev["step"]
                position = (step - prev["step"]) / span
                if curr["interpolation"] == "cosine":
                    position = 0.5 * (1 - math.cos(math.pi * position))
                return prev["lr"] + (curr["lr"] - prev["lr"]) * position

        return self.milestones[-1]["lr"]

class ExponentialWarmupThenPiecewiseLR(torch.optim.lr_scheduler._LRScheduler):
    """Use ExponentialWarmupPlateauLR until a cutoff, then absolute piecewise LR."""

    def __init__(
            self,
            optimizer,
            warmup=0.,
            warmup_steps=0,
            warmup_until_step=0,
            milestones=None,
            last_epoch=-1
    ):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        if warmup_steps < 0:
            raise ValueError('Invalid value for warmup_steps')
        if warmup_until_step < 0:
            raise ValueError('Invalid value for warmup_until_step')
        if not milestones:
            raise ValueError("milestones must contain at least one milestone")

        parsed_milestones = []
        for milestone in milestones:
            step = int(milestone["step"])
            lr = float(milestone["lr"])
            interpolation = milestone.get("interpolation", "linear")
            if step <= warmup_until_step:
                raise ValueError("milestone steps must be > warmup_until_step")
            if lr < 0:
                raise ValueError("milestone lr must be >= 0")
            if interpolation not in {"linear", "cosine"}:
                raise ValueError("interpolation must be 'linear' or 'cosine'")
            parsed_milestones.append({
                "step": step,
                "lr": lr,
                "interpolation": interpolation,
            })

        parsed_milestones = sorted(parsed_milestones, key=lambda item: item["step"])
        for prev, curr in zip(parsed_milestones, parsed_milestones[1:]):
            if curr["step"] <= prev["step"]:
                raise ValueError("milestone steps must be strictly increasing")

        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.warmup_until_step = warmup_until_step
        self.milestones = parsed_milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        step = max(self.last_epoch, 0)
        if step <= self.warmup_until_step:
            return [base_lr * self._warmup_mult(step) for base_lr in self.base_lrs]

        warmup_lrs = [
            base_lr * self._warmup_mult(self.warmup_until_step)
            for base_lr in self.base_lrs
        ]
        return [
            self._piecewise_lr_value(step, warmup_lr)
            for warmup_lr in warmup_lrs
        ]

    def _warmup_mult(self, step):
        if self.warmup == 0:
            return 1.

        if step >= self.warmup_steps:
            return 1.

        warmup = 1 - self.warmup ** (step + 1)
        target = 1 - self.warmup ** (self.warmup_steps + 1)
        return min(1., warmup / target)

    def _piecewise_lr_value(self, step, warmup_lr):
        prev_step = self.warmup_until_step
        prev_lr = warmup_lr

        for curr in self.milestones:
            if step <= curr["step"]:
                span = curr["step"] - prev_step
                position = (step - prev_step) / span
                if curr["interpolation"] == "cosine":
                    position = 0.5 * (1 - math.cos(math.pi * position))
                return prev_lr + (curr["lr"] - prev_lr) * position

            prev_step = curr["step"]
            prev_lr = curr["lr"]

        return self.milestones[-1]["lr"]

def create_optimizer_from_config(optimizer_config, parameters):
    """Create optimizer from config.

    Args:
        parameters (iterable): parameters to optimize.
        optimizer_config (dict): optimizer config.

    Returns:
        torch.optim.Optimizer: optimizer.
    """

    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config, optimizer):
    """Create scheduler from config.

    Args:
        scheduler_config (dict): scheduler config.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: scheduler.
    """
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    elif scheduler_config["type"] == "InverseLRHalfMix":
        scheduler_fn = InverseLRHalfMix
    elif scheduler_config["type"] == "ExponentialWarmupPlateauLR":
        scheduler_fn = ExponentialWarmupPlateauLR
    elif scheduler_config["type"] == "WarmupHoldCosineFloorLR":
        scheduler_fn = WarmupHoldCosineFloorLR
    elif scheduler_config["type"] == "PiecewiseLRScheduler":
        scheduler_fn = PiecewiseLRScheduler
    elif scheduler_config["type"] == "ExponentialWarmupThenPiecewiseLR":
        scheduler_fn = ExponentialWarmupThenPiecewiseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler

def logger_project_name(logger) -> str:
    if isinstance(logger, WandbLogger):
        return logger.experiment.project
    elif isinstance(logger, CometLogger):
        return logger.name

def log_metric(logger, key, value, step=None):
    # Route scalar metrics through the logger interface so external logger
    # implementations can manage step bookkeeping consistently.
    if isinstance(logger, (WandbLogger, CometLogger)):
        logger.log_metrics({key: value}, step=step)

def log_audio(logger, key, audio_path, sample_rate, caption=None, step=None):
    if isinstance(logger, WandbLogger):
        payload = {key: wandb.Audio(audio_path, sample_rate=sample_rate, caption=caption)}
        if step is not None:
            payload["trainer/global_step"] = step
        logger.experiment.log(payload)
    elif isinstance(logger, CometLogger):
        logger.experiment.log_audio(audio_path, file_name=key, sample_rate=sample_rate)

def log_image(logger, key, img_data, step=None):
    if isinstance(logger, WandbLogger):
        payload = {key: wandb.Image(img_data)}
        if step is not None:
            payload["trainer/global_step"] = step
        logger.experiment.log(payload)
    elif isinstance(logger, CometLogger):
        logger.experiment.log_image(img_data, name=key)

def log_point_cloud(logger, key, tokens, caption=None, step=None):
    if isinstance(logger, WandbLogger):
        point_cloud = pca_point_cloud(tokens)
        payload = {key: point_cloud}
        if step is not None:
            payload["trainer/global_step"] = step
        logger.experiment.log(payload)
    elif isinstance(logger, CometLogger):
        point_cloud = pca_point_cloud(tokens, rgb_float=True, output_type="points")
        #logger.experiment.log_points_3d(scene_name=key, points=point_cloud)
