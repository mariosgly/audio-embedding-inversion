import pytorch_lightning as pl
import gc
import hashlib
import random
import torch
import torchaudio
import typing as tp
import tempfile
import os
import gc
import soundfile as sf
import wandb

import auraloss
from ema_pytorch import EMA
from einops import rearrange
from functools import partial
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pytorch_lightning as pl
import torch


from ..interface.aeiou import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, sample_flow_pingpong, truncated_logistic_normal_rescaled, DistributionShift, sample_timesteps_logsnr
from ..models.diffusion import DiffusionModelWrapper, ConditionedDiffusionModelWrapper
from ..models.autoencoders import DiffusionAutoencoder
from ..models.inpainting import random_inpaint_mask
from ..models.lora import add_lora, get_lora_layers, get_lora_params, get_lora_state_dict, LoRAParametrization, prepare_dora_state_dict, resolve_adapter_type, save_lora_safetensors
from .autoencoders import create_loss_modules_from_bottleneck
from .losses import AuralossLoss, L1Loss, MSELoss, MultiLoss
from .losses import auraloss as loss_auraloss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_point_cloud

from time import time




class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


def trim_to_shortest(a, b):
    if a.shape[-1] > b.shape[-1]:
        return a[..., :b.shape[-1]], b
    if b.shape[-1] > a.shape[-1]:
        return a, b[..., :a.shape[-1]]
    return a, b

class DiffusionUncondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training an unconditional audio diffusion model (like Dance Diffusion).
    '''
    def __init__(
            self,
            model: DiffusionModelWrapper,
            lr: float = 1e-4,
            pre_encoded: bool = False
    ):
        super().__init__()

        self.diffusion = model

        self.diffusion_ema = EMA(
            self.diffusion.model,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                     "targets",
                     weight=1.0,
                     name="mse_loss"
                )
        ]

        self.losses = MultiLoss(loss_modules)

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        loss_info = {}

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        if self.diffusion.pretransform is not None:
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        loss_info["reals"] = diffusion_input

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas

        with torch.amp.autocast("cuda"):
            v = self.diffusion(noised_inputs, t)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):

        self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionUncondDemoCallback(pl.Callback):
    def __init__(self,
                 demo_every=2000,
                 num_demos=8,
                 demo_steps=250,
                 sample_rate=48000
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        demo_samples = module.diffusion.sample_size

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            with torch.amp.autocast("cuda"):
                fakes = sample(module.diffusion_ema, noise, self.demo_steps, 0)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            log_audio(
                trainer.logger, "demo", filename,
                sample_rate=self.sample_rate, caption='Reconstructed')
            log_image(
                trainer.logger, "demo_melspec_left",
                audio_spectrogram_image(fakes))

            del fakes
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            
def gpu_mem(tag):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[mem] {tag}: alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G")


class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal", "log_snr"] = "uniform",
            timestep_sampler_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            p_one_shot: float = 0.0,
            inpainting_config: dict = None,
            loss_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
            memory_debug: bool = False,
            memory_debug_first_n_steps: int = 1,
            lora_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
            lora_state_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
            lora_full_train_param_patterns: tp.Optional[tp.List[str]] = None,
            trainable_param_patterns: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        super().__init__()

        self.diffusion = model

        self.lora_config = lora_config
        self.lora_full_train_param_patterns = lora_full_train_param_patterns or []
        self.trainable_param_patterns = trainable_param_patterns

        if self.lora_config is not None:
            use_ema = False
            self.diffusion.model.eval().requires_grad_(False)
            self.diffusion.conditioner.eval().requires_grad_(False)

            rank = self.lora_config.get("rank", 8)
            lora_alpha = self.lora_config.get("alpha", rank)
            adapter_type = resolve_adapter_type(self.lora_config.get("adapter_type", "lora"), lora_state_dict)
            include = self.lora_config.get("include", None)
            exclude = self.lora_config.get("exclude", None)
            print(f"LoRA config: rank={rank}, alpha={lora_alpha}, adapter_type={adapter_type}")
            if include:
                print(f"  include: {include}")
            if exclude:
                print(f"  exclude: {exclude}")
            if adapter_type.endswith("-xs"):
                print("WARNING: LoRA-XS without svd_bases_path will compute SVD bases per adapted layer at startup")

            adapter_config = {
                torch.nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha, adapter_type=adapter_type),
                },
                torch.nn.Conv1d: {
                    "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=lora_alpha, adapter_type=adapter_type),
                },
            }
            add_lora(self.diffusion.model, adapter_config, include=include, exclude=exclude)
            add_lora(self.diffusion.conditioner, adapter_config, include=include, exclude=exclude)
            print("LoRA layers:", len(get_lora_layers(self.diffusion)))

            if lora_state_dict is not None:
                prepare_dora_state_dict(lora_state_dict)
                self.diffusion.model.load_state_dict(lora_state_dict, strict=False)
                self.diffusion.conditioner.load_state_dict(lora_state_dict, strict=False)

            self._apply_lora_full_train_param_patterns(self.lora_full_train_param_patterns)

        if self.trainable_param_patterns is not None and self.lora_config is None:
            self._apply_trainable_param_patterns(self.trainable_param_patterns)

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.timestep_sampler_options = {} if timestep_sampler_options is None else timestep_sampler_options

        if self.timestep_sampler == "log_snr":
            self.mean_logsnr = self.timestep_sampler_options.get("mean_logsnr", -1.2)
            self.std_logsnr = self.timestep_sampler_options.get("std_logsnr", 2.0)

        self.p_one_shot = p_one_shot

        self.diffusion_objective = model.diffusion_objective
        self.memory_debug = memory_debug
        self.memory_debug_first_n_steps = memory_debug_first_n_steps

        self.loss_modules = self._build_loss_modules(loss_config)

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        # Inpainting
        self.inpainting_config = inpainting_config
        
        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})

        # Validation
        self.validation_timesteps = validation_timesteps

    def _matches_param_pattern(self, name: str, patterns: tp.Optional[tp.List[str]]) -> bool:
        if not patterns:
            return False
        return any(pattern in name for pattern in patterns)

    def _apply_lora_full_train_param_patterns(self, patterns: tp.List[str]):
        if not patterns:
            return

        matched = []
        for name, param in self.diffusion.named_parameters():
            if self._matches_param_pattern(name, patterns):
                param.requires_grad = True
                matched.append(name)

        print(f"LoRA full-train parameter patterns: {len(matched)} tensors trainable outside adapters")
        for name in matched[:80]:
            print(f"  FULL_TRAIN: {name}")
        if len(matched) > 80:
            print(f"  ... {len(matched) - 80} more full-train tensors")

    def _get_lora_full_train_params(self):
        if not self.lora_full_train_param_patterns:
            return []
        params = []
        seen = set()
        for name, param in self.diffusion.named_parameters():
            if self._matches_param_pattern(name, self.lora_full_train_param_patterns):
                ident = id(param)
                if ident not in seen:
                    params.append(param)
                    seen.add(ident)
        return params

    def _get_lora_full_train_state_dict(self):
        if not self.lora_full_train_param_patterns:
            return {}
        return {
            key: value
            for key, value in self.diffusion.state_dict().items()
            if self._matches_param_pattern(key, self.lora_full_train_param_patterns)
        }

    def _apply_trainable_param_patterns(self, config: tp.Dict[str, tp.Any]):
        include = config.get("include", [])
        exclude = config.get("exclude", [])
        freeze_unmatched = config.get("freeze_unmatched", True)

        matched = []
        for name, param in self.diffusion.named_parameters():
            trainable = self._matches_param_pattern(name, include)
            if self._matches_param_pattern(name, exclude):
                trainable = False
            if freeze_unmatched:
                param.requires_grad = trainable
            elif trainable:
                param.requires_grad = True
            if param.requires_grad:
                matched.append(name)

        print(
            f"Trainable parameter pattern mode: {len(matched)} tensors trainable "
            f"(freeze_unmatched={freeze_unmatched})"
        )
        for name in matched[:80]:
            print(f"  TRAINABLE: {name}")
        if len(matched) > 80:
            print(f"  ... {len(matched) - 80} more trainable tensors")

    def _build_loss_modules(self, loss_config: tp.Optional[tp.Dict[str, tp.Any]]):
        if loss_config is None:
            loss_config = {
                "mse": {
                    "weight": 1.0,
                }
            }

        loss_modules = []

        mse_config = loss_config.get("mse", {"weight": 1.0})
        mse_weight = mse_config.get("weight", 1.0)
        if mse_weight > 0:
            loss_modules.append(
                MSELoss(
                    "output",
                    "targets",
                    weight=mse_weight,
                    mask_key="padding_mask" if self.mask_padding else None,
                    name="mse_loss",
                    decay=mse_config.get("decay", 1.0),
                )
            )

        waveform_l1_config = loss_config.get("waveform_l1")
        if waveform_l1_config is not None and waveform_l1_config.get("weight", 0.0) > 0:
            self._require_pretransform_loss("waveform_l1")
            loss_modules.append(
                L1Loss(
                    "pred_audio",
                    "audio_reals_for_loss",
                    weight=waveform_l1_config["weight"],
                    name="waveform_l1_loss",
                    decay=waveform_l1_config.get("decay", 1.0),
                )
            )

        ae_reencode_config = loss_config.get("ae_reencode_latent_l1")
        if ae_reencode_config is not None and ae_reencode_config.get("weight", 0.0) > 0:
            self._require_pretransform_loss("ae_reencode_latent_l1")
            loss_modules.append(
                L1Loss(
                    "pred_reencoded_latents",
                    "reals",
                    weight=ae_reencode_config["weight"],
                    name="ae_reencode_latent_l1_loss",
                    decay=ae_reencode_config.get("decay", 1.0),
                )
            )

        mrstft_config = loss_config.get("mrstft")
        if mrstft_config is not None and mrstft_config.get("weight", 0.0) > 0:
            self._require_pretransform_loss("mrstft")
            mrstft_kwargs = {
                "sample_rate": self.diffusion.sample_rate,
            }
            mrstft_kwargs.update(mrstft_config.get("config", {}))
            self.mrstft = loss_auraloss.MultiResolutionSTFTLoss(**mrstft_kwargs)
            loss_modules.append(
                AuralossLoss(
                    self.mrstft,
                    input_key="pred_audio",
                    target_key="audio_reals_for_loss",
                    name="mrstft_loss",
                    weight=mrstft_config["weight"],
                    decay=mrstft_config.get("decay", 1.0),
                )
            )

        if not loss_modules:
            raise ValueError("At least one diffusion loss must be enabled")

        return loss_modules

    def _require_pretransform_loss(self, loss_name: str):
        if self.diffusion.pretransform is None:
            raise ValueError(f"{loss_name} requires model.pretransform to be configured")

    def _predict_denoised_latents(self, noised_inputs, model_output, alphas, sigmas):
        if self.diffusion_objective == "v":
            return alphas * noised_inputs - sigmas * model_output
        if self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            return noised_inputs - sigmas * model_output
        raise ValueError(f"Unsupported diffusion objective for reconstruction losses: {self.diffusion_objective}")
    
    def _memory_debug_enabled(self):
        return self.memory_debug and torch.cuda.is_available() and self.global_step < self.memory_debug_first_n_steps

    def _log_gpu_memory(self, tag: str):
        if not self._memory_debug_enabled():
            return

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        print(f"[gpu-mem] step={self.global_step} {tag}: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB peak={peak:.2f} GiB")

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']

        if self.lora_config is not None:
            opt_params = [
                *get_lora_params(self.diffusion.model),
                *get_lora_params(self.diffusion.conditioner),
                *self._get_lora_full_train_params(),
            ]
        elif self.trainable_param_patterns is not None:
            opt_params = [p for p in self.diffusion.parameters() if p.requires_grad]
        else:
            opt_params = self.diffusion.parameters()

        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], opt_params)

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        


        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals
        
        if self._memory_debug_enabled():
            torch.cuda.reset_peak_memory_stats(self.device)
            self._log_gpu_memory("start")

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        #with torch.amp.autocast(device_type="cuda"):
        conditioning = self.diffusion.conditioner(metadata, self.device)

        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # Check for wrapped padding masks to avoid interpolation error
        first_padding_mask = metadata[0]["padding_mask"]
        if isinstance(first_padding_mask, list) and len(first_padding_mask) == 1:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
        else:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)


        
        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast("cuda"), torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    self._log_gpu_memory("after_pretransform_encode")
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)

            # Flip the distribution
            t = 1 - t
        elif self.timestep_sampler == "log_snr":
            t = sample_timesteps_logsnr(reals.shape[0], mean_logsnr=self.mean_logsnr, std_logsnr=self.std_logsnr).to(self.device)
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            # Shift the distribution
            t = self.diffusion.dist_shift.time_shift(t, diffusion_input.shape[2])

        if self.p_one_shot > 0:
            # Set t to 1 with probability p_one_shot
            t = torch.where(torch.rand_like(t) < self.p_one_shot, torch.ones_like(t), t)
       

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            targets = noise - diffusion_input
            


        p.tick("noise")

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        if self.inpainting_config is not None:

            # Max mask size is the full sequence length
            max_mask_length = diffusion_input.shape[2]

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
        self._log_gpu_memory("after_diffusion_forward")


        p.tick("diffusion")

        loss_info.update({
            "output": output,
            "targets": targets,
            "padding_mask": padding_masks if use_padding_mask else None,
        })

        configured_loss_names = {loss_module.name for loss_module in self.loss_modules}
        if (
            "waveform_l1_loss" in configured_loss_names
            or "ae_reencode_latent_l1_loss" in configured_loss_names
            or "mrstft_loss" in configured_loss_names
        ):
            with torch.amp.autocast("cuda"):
                self._log_gpu_memory("before_predicting_latents")
                pred_latents = self._predict_denoised_latents(noised_inputs, output, alphas, sigmas)
                self._log_gpu_memory("after_predicting_latents")

                pred_audio = self.diffusion.pretransform.decode(pred_latents)
                self._log_gpu_memory("after_pretransform_decode")


                if self.pre_encoded:
                        audio_reals_for_loss = self.diffusion.pretransform.decode(diffusion_input)
                else:
                    audio_reals_for_loss = reals

                pred_audio, audio_reals_for_loss = trim_to_shortest(pred_audio, audio_reals_for_loss)

                loss_info["pred_audio"] = pred_audio
                loss_info["audio_reals_for_loss"] = audio_reals_for_loss

                if "ae_reencode_latent_l1_loss" in configured_loss_names:
                    self._log_gpu_memory("before_predicting_latents_of_new_wave")
                    pred_reencoded_latents = self.diffusion.pretransform.encode(pred_audio)
                    pred_reencoded_latents, target_latents = trim_to_shortest(pred_reencoded_latents, diffusion_input)
                    loss_info["pred_reencoded_latents"] = pred_reencoded_latents
                    loss_info["reals"] = target_latents
                    self._log_gpu_memory("after_pretransform_reencode")

        loss, losses = self.losses(loss_info)
        self._log_gpu_memory("after_total_loss")

        p.tick("loss")
        
        # print("\n" + "="*60)
        # print(f"=== TRAINING STEP (Batch {batch_idx}) ===")
        # print(f"1. Raw Audio Input (reals) shape: {reals.shape}")
        # print("2. Conditioning Tensors Generated:")
        # for k, v in conditioning.items():
        #     if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
        #         print(f"   - {k}: {v[0].shape} (list of tensors)")
        #     elif isinstance(v, torch.Tensor):
        #         print(f"   - {k}: {v.shape}")
        # print(f"3. Padding Masks shape: {padding_masks.shape}")
        # print(f"4. Compressed Latents (after VAE pretransform) shape: {diffusion_input.shape}")
        # print(f"5. Sampled Timesteps (t) shape: {t.shape}")
        # print(f"6. Added Noise shape: {noise.shape}")
        # print(f"7. Noised Inputs (x -> going to model) shape: {noised_inputs.shape}")
        # print(f"8. Training Targets shape: {targets.shape}") 
        # print(f"9. Diffusion Model Output shape: {output.shape}")
        # print("="*60 + "\n")

        if self.log_loss_info:
            # Loss debugging logs
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none")

            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

            # gather loss_all across all GPUs
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

            # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)


        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    @staticmethod
    def _validation_noise_seed(sample_metadata, validation_timestep):
        stable_parts = [
            sample_metadata.get("path", ""),
            sample_metadata.get("relpath", ""),
            sample_metadata.get("timestamps", ""),
            sample_metadata.get("seconds_start", ""),
            sample_metadata.get("seconds_total", ""),
            f"{float(validation_timestep):.8f}",
        ]
        key = "|".join(str(part) for part in stable_parts)
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)

    def _deterministic_validation_noise(self, diffusion_input, metadata, validation_timestep):
        sample_shape = tuple(diffusion_input.shape[1:])
        noise = []
        for sample_metadata in metadata:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._validation_noise_seed(sample_metadata, validation_timestep))
            noise.append(torch.randn(sample_shape, generator=generator, dtype=torch.float32))

        return torch.stack(noise, dim=0).to(device=diffusion_input.device, dtype=diffusion_input.dtype)

    def validation_step(self, batch, batch_idx):

        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        batch_size = reals.shape[0]

        diffusion_input = reals
        avg_val_losses = []

        with torch.amp.autocast("cuda"), torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)
        # TODO: decide what to do with padding masks during validation

        # # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        # use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # # Create batch tensor of attention masks from the "mask" field of the metadata array
        # if use_padding_mask:
        #     padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast("cuda"), torch.no_grad():
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    # # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    # if use_padding_mask:
                    #     padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                alphas, sigmas = 1-t, t

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = self._deterministic_validation_noise(diffusion_input, metadata, validation_timestep)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                targets = noise - diffusion_input

            extra_args = {}

            # if use_padding_mask:
            #     extra_args["mask"] = padding_masks

            with torch.amp.autocast("cuda"), torch.no_grad():
                output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0, **extra_args)

                # print(f"   -> Timestep {validation_timestep:.1f} | Noised Input: {noised_inputs.shape} | Output: {output.shape}")
                val_loss = F.mse_loss(output, targets)
                metric_name = f'val/loss_{validation_timestep:.1f}'
                self.log(
                    metric_name,
                    val_loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                avg_val_losses.append(val_loss)
                
                
        # print(f"=== VALIDATION STEP (Batch {batch_idx}) ===")
        # print(f"1. Val Raw Audio Input shape: {reals.shape}")
        # print("2. Val Conditioning Tensors Generated:")
        # for k, v in conditioning.items():
        #     if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
        #         print(f"   - {k}: {v[0].shape} (list of tensors)")
        #     elif isinstance(v, torch.Tensor):
        #         print(f"   - {k}: {v.shape}")
        # print(f"3. Val Latents (after VAE encode) shape: {diffusion_input.shape}")        
        # print(f"4. Running validation timesteps: {self.validation_timesteps}")
        # print("="*60 + "\n")

        if avg_val_losses:
            self.log(
                'val/avg_loss',
                torch.stack(avg_val_losses).mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

    def export_lora_safetensors(self, path):
        if self.lora_config is None:
            raise ValueError("No LoRA config -- this wrapper is not in LoRA mode")
        state_dict = {
            **get_lora_state_dict(self.diffusion.model),
            **get_lora_state_dict(self.diffusion.conditioner),
            **self._get_lora_full_train_state_dict(),
        }
        save_lora_safetensors(state_dict, self.lora_config, path)

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is not None:
            checkpoint.clear()
            checkpoint["state_dict"] = {
                **get_lora_state_dict(self.diffusion.model),
                **get_lora_state_dict(self.diffusion.conditioner),
                **self._get_lora_full_train_state_dict(),
            }
            checkpoint["lora_config"] = self.lora_config
            checkpoint["lora_full_train_param_patterns"] = self.lora_full_train_param_patterns


class DiffusionCondDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl=None,
        demo_every=2000,
        num_demos=8,
        sample_size=65536,
        demo_steps=250,
        sample_rate=48000,
        demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = None,
        demo_cfg_scales: tp.Optional[tp.List[int]] = None,
        demo_cond_from_batch: bool = False,
        display_audio_cond: bool = False,
        cond_display_configs: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
    ):
        super().__init__()

        self.demo_dl = demo_dl
        self.demo_dl_iter = iter(demo_dl) if demo_dl is not None else None

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning or {}
        self.demo_cfg_scales = demo_cfg_scales or [3, 5, 7]

        self.demo_cond_from_batch = demo_cond_from_batch
        self.display_audio_cond = display_audio_cond
        self.cond_display_configs = cond_display_configs

    @staticmethod
    def _prepare_audio_for_demo(x: torch.Tensor) -> torch.Tensor:
        """
        Prepare waveform for saving.
        Expects [C, T] and returns float32 CPU tensor in [-1, 1].
        """
        x = x.detach().to(torch.float32)
        x = x.clamp(-1, 1).cpu()
        return x

    @staticmethod
    def _prepare_audio_for_spectrogram(x: torch.Tensor) -> torch.Tensor:
        """
        Prepare waveform for audio_spectrogram_image using the old repo convention.
        Expects [C, T] and returns int16 CPU tensor scaled to +/-32767.
        Loudness faithfulness is not preserved; this is for visualization only.
        """
        x = x.detach().to(torch.float32)
        max_abs = torch.max(torch.abs(x))
        if max_abs.item() == 0:
            max_abs = max_abs + 1e-8
        x = x.div(max_abs).mul(32767).to(torch.int16).cpu()
        return x

    @staticmethod
    def _save_wav(path: str, x: torch.Tensor, sample_rate: int) -> None:
        """
        Save [C, T] waveform tensor using soundfile as PCM16 WAV.
        soundfile expects [T, C].
        """
        x = DiffusionCondDemoCallback._prepare_audio_for_demo(x)
        sf.write(path, x.transpose(0, 1).numpy(), sample_rate, subtype="PCM_16")

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: "DiffusionCondTrainingWrapper", outputs, batch, batch_idx):
        from pytorch_lightning.loggers import WandbLogger

        if self.demo_every is None or self.demo_every <= 0 or self.num_demos <= 0:
            return
        
        
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()
        temp_audio_files = []
        try:
            print(f"\n--- Generating Train & Test Demos (Step {trainer.global_step}) ---")
            self.last_demo_step = trainer.global_step
            wandb_payload = {} if isinstance(trainer.logger, WandbLogger) else None

            train_reals, train_cond = batch
            self._generate_demo_batch(
                trainer,
                module,
                train_reals,
                train_cond,
                log_mels=True,
                prefix="train_demo",
                debug=True,
                wandb_payload=wandb_payload,
                temp_audio_files=temp_audio_files,
            )
            print("finished generating train demo")

            if self.demo_dl is not None:
                try:
                    test_reals, test_cond = next(self.demo_dl_iter)
                except StopIteration:
                    self.demo_dl_iter = iter(self.demo_dl)
                    test_reals, test_cond = next(self.demo_dl_iter)

                self._generate_demo_batch(
                    trainer,
                    module,
                    test_reals,
                    test_cond,
                    log_mels=False,
                    prefix="test_demo",
                    debug=True,
                    wandb_payload=wandb_payload,
                    temp_audio_files=temp_audio_files,
                )
                print("finished generating test demo")

            if wandb_payload:
                wandb_payload["trainer/global_step"] = trainer.global_step
                trainer.logger.experiment.log(wandb_payload)

        finally:
            for filename in temp_audio_files:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
            module.train()

    def _generate_demo_batch(
        self,
        trainer,
        module,
        reals,
        cond,
        prefix: str,
        log_mels: bool = True,
        debug: bool = False,
        wandb_payload=None,
        temp_audio_files=None,
    ):
        def dprint(msg: str):
            if debug:
                print(f"[DEMO_DEBUG][{prefix}][step={trainer.global_step}] {msg}", flush=True)

        def queue_audio(key: str, filename: str, sample_rate: int, caption=None):
            if wandb_payload is not None:
                wandb_payload[key] = wandb.Audio(filename, sample_rate=sample_rate, caption=caption)
                if temp_audio_files is not None:
                    temp_audio_files.append(filename)
            else:
                log_audio(trainer.logger, key, filename, sample_rate, caption=caption, step=trainer.global_step)
                os.remove(filename)

        def queue_image(key: str, img_data):
            if wandb_payload is not None:
                wandb_payload[key] = wandb.Image(img_data)
            else:
                log_image(trainer.logger, key, img_data, step=trainer.global_step)

        try:
            if isinstance(reals, torch.Tensor) and reals.ndim == 4 and reals.shape[0] == 1:
                dprint("squeezing webdataset dim: reals[0]")
                reals = reals[0]

            if not isinstance(reals, torch.Tensor):
                raise TypeError(f"Expected reals Tensor, got {type(reals)}")

            reals = reals[: self.num_demos].to(module.device)
            if isinstance(cond, (list, tuple)):
                cond = cond[: self.num_demos]

            demo_cond = cond
            if (not self.demo_cond_from_batch) and self.demo_conditioning:
                demo_cond = self.demo_conditioning
                dprint("using self.demo_conditioning (override)")

            demo_samples = self.demo_samples
            if module.diffusion.pretransform is not None:
                demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

            batch_size = reals.shape[0]
            noise = torch.randn(
                [batch_size, module.diffusion.io_channels, demo_samples],
                device=module.device,
            )

            with torch.amp.autocast("cuda"):
                conditioning = module.diffusion.conditioner(demo_cond, module.device)
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            # 1) Optional raw audio conditioning display
            if self.display_audio_cond:
                if not isinstance(demo_cond, (list, tuple)):
                    dprint("display_audio_cond=True but demo_cond not list/tuple; skipping")
                else:
                    audio_inputs = torch.cat([c["audio"] for c in demo_cond], dim=0)
                    audio_inputs = rearrange(audio_inputs, "b d n -> d (b n)")

                    audio_save = self._prepare_audio_for_demo(audio_inputs)
                    audio_spec = self._prepare_audio_for_spectrogram(audio_inputs)

                    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    filename = f.name
                    f.close()

                    self._save_wav(filename, audio_save, self.sample_rate)
                    queue_audio(f"{prefix}_audio_cond", filename, self.sample_rate)

                    if log_mels:
                        queue_image(
                            f"{prefix}_audio_cond_melspec_left",
                            audio_spectrogram_image(audio_spec),
                        )

            # 2) Pre-generation conditioning display
            if self.cond_display_configs is not None:
                if not isinstance(demo_cond, (list, tuple)):
                    dprint("cond_display_configs set but demo_cond not list/tuple; skipping")
                else:
                    for cond_display_config in self.cond_display_configs:
                        cond_id = cond_display_config.get("id", None)
                        cond_type = cond_display_config.get("type", None)

                        if cond_type != "audio":
                            continue

                        audio_cond_config = cond_display_config.get("config", {})
                        is_pre_encoded = audio_cond_config.get("pre_encoded", False)

                        audio_inputs = torch.stack([c[cond_id] for c in demo_cond], dim=0)

                        if is_pre_encoded:
                            audio_inputs = module.diffusion.pretransform.decode(audio_inputs)

                        audio_inputs = rearrange(audio_inputs, "b d n -> d (b n)")

                        audio_save = self._prepare_audio_for_demo(audio_inputs)
                        audio_spec = self._prepare_audio_for_spectrogram(audio_inputs)

                        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        filename = f.name
                        f.close()

                        self._save_wav(filename, audio_save, self.sample_rate)
                        queue_audio(f"{prefix}_{cond_id}", filename, self.sample_rate)

                        if log_mels:
                            queue_image(
                                f"{prefix}_{cond_id}_melspec_left",
                                audio_spectrogram_image(audio_spec),
                            )

            # 3) Ground truth logging
            if module.diffusion.pretransform is not None and getattr(module, "pre_encoded", False):
                reals_audio = module.diffusion.pretransform.decode(reals)
            else:
                reals_audio = reals

            gt_audio = rearrange(reals_audio, "b d n -> d (b n)")
            gt_save = self._prepare_audio_for_demo(gt_audio)
            gt_spec = self._prepare_audio_for_spectrogram(gt_audio)

            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            gt_filename = f.name
            f.close()

            self._save_wav(gt_filename, gt_save, self.sample_rate)
            queue_audio(f"{prefix}_ground_truth", gt_filename, self.sample_rate)

            if log_mels:
                queue_image(
                    f"{prefix}_ground_truth_melspec_left",
                    audio_spectrogram_image(gt_spec),
                )

            # 4) Generate + log fakes
            for cfg_scale in self.demo_cfg_scales:
                print(f"[{prefix}] Generating demo for cfg scale {cfg_scale}", flush=True)

                with torch.amp.autocast("cuda"):
                    model = (
                        module.diffusion_ema.ema_model
                        if module.diffusion_ema is not None
                        else module.diffusion.model
                    )

                    if module.diffusion_objective == "v":
                        fakes = sample(
                            model,
                            noise,
                            self.demo_steps,
                            0,
                            **cond_inputs,
                            cfg_scale=cfg_scale,
                            dist_shift=module.diffusion.dist_shift,
                            batch_cfg=True,
                        )
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(
                            model,
                            noise,
                            self.demo_steps,
                            **cond_inputs,
                            cfg_scale=cfg_scale,
                            dist_shift=module.diffusion.dist_shift,
                            batch_cfg=True,
                        )
                    elif module.diffusion_objective == "rf_denoiser":
                        logsnr = torch.linspace(-6, 2, self.demo_steps + 1, device=module.device)
                        sigmas = torch.sigmoid(-logsnr)
                        sigmas[0] = 1.0
                        sigmas[-1] = 0.0

                        fakes = sample_flow_pingpong(
                            model,
                            noise,
                            sigmas=sigmas,
                            **cond_inputs,
                            cfg_scale=cfg_scale,
                            dist_shift=module.diffusion.dist_shift,
                            batch_cfg=True,
                        )
                    else:
                        raise ValueError(f"Unknown diffusion_objective: {module.diffusion_objective}")

                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                fakes = rearrange(fakes, "b d n -> d (b n)")
                fakes_save = self._prepare_audio_for_demo(fakes)
                fakes_spec = self._prepare_audio_for_spectrogram(fakes)

                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                filename = f.name
                f.close()

                self._save_wav(filename, fakes_save, self.sample_rate)
                queue_audio(f"{prefix}_cfg_{cfg_scale}", filename, self.sample_rate)

                if log_mels:
                    queue_image(
                        f"{prefix}_melspec_left_cfg_{cfg_scale}",
                        audio_spectrogram_image(fakes_spec),
                    )

                del fakes
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed generating {prefix}: {type(e).__name__}: {e}", flush=True)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

class DiffusionCondInpaintDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000,
        num_demos=8,
        demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7]
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.demo_cfg_scales = demo_cfg_scales
        self.num_demos = num_demos
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        try:

            demo_reals, metadata = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit to num_demos
            demo_reals = demo_reals[:self.num_demos]
            metadata = metadata[:self.num_demos]

            demo_reals = demo_reals.to(module.device)

            if not module.pre_encoded:
                # Log the real audio
                log_image(trainer.logger, f'demo_reals_melspec_left', audio_spectrogram_image(rearrange(demo_reals, "b d n -> d (b n)").mul(32767).to(torch.int16).cpu()))

                if module.diffusion.pretransform is not None:
                    module.diffusion.pretransform.to(module.device)
                    demo_reals = module.diffusion.pretransform.encode(demo_reals)
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(module.diffusion.pretransform, "scale") and module.diffusion.pretransform.scale != 1.0:
                    demo_reals = demo_reals / module.diffusion.pretransform.scale

            demo_samples = demo_reals.shape[2]

            # Get conditioning
            conditioning = module.diffusion.conditioner(metadata, module.device)

            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(module.device) # Shape (batch_size, sequence_length)

            masked_input, mask = random_inpaint_mask(demo_reals, padding_masks=padding_masks, **module.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [masked_input]

            if module.diffusion.pretransform is not None:
                log_image(trainer.logger, f'demo_masked_input', tokens_spectrogram_image(masked_input.cpu()))
            else:
                log_image(trainer.logger, f'demo_masked_input', audio_spectrogram_image(rearrange(masked_input, "b c t -> c (b t)").mul(32767).to(torch.int16).cpu()))

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            noise = torch.randn([demo_reals.shape[0], module.diffusion.io_channels, demo_samples]).to(module.device)

            # Cast the noise to the dtype of the model
            model_dtype = next(module.diffusion.parameters()).dtype
            noise = noise.to(model_dtype)

            for cfg_scale in self.demo_cfg_scales:
                model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model
                print(f"Generating demo for cfg scale {cfg_scale}")

                with torch.amp.autocast("cuda"):
                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rf_denoiser":
                        logsnr = torch.linspace(-6, 2, self.demo_steps+1).to(module.device)
                        sigmas = torch.sigmoid(-logsnr)

                        sigmas[0] = 1.0
                        sigmas[-1] = 0.0

                        fakes = sample_flow_pingpong(model, noise, sigmas=sigmas, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_audio(trainer.logger, f'demo_cfg_{cfg_scale}', filename, self.sample_rate)
                log_image(trainer.logger, f'demo_melspec_left_cfg_{cfg_scale}', audio_spectrogram_image(fakes))

        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e

class DiffusionAutoencoderTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a diffusion autoencoder
    '''
    def __init__(
            self,
            model: DiffusionAutoencoder,
            lr: float = 1e-4,
            ema_copy = None,
            use_reconstruction_loss: bool = False
    ):
        super().__init__()

        self.diffae = model

        self.diffae_ema = EMA(
            self.diffae,
            ema_model=ema_copy,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                    "targets",
                    weight=1.0,
                    name="mse_loss"
            )
        ]

        if model.bottleneck is not None:
            # TODO: Use loss config for configurable bottleneck weights and reconstruction losses
            loss_modules += create_loss_modules_from_bottleneck(model.bottleneck, {})

        self.use_reconstruction_loss = use_reconstruction_loss

        if use_reconstruction_loss:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            sample_rate = model.sample_rate

            stft_loss_args = {
                "fft_sizes": scales,
                "hop_sizes": hop_sizes,
                "win_lengths": win_lengths,
                "perceptual_weighting": True
            }

            out_channels = model.out_channels

            if model.pretransform is not None:
                out_channels = model.pretransform.io_channels

            if out_channels == 2:
                self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

            loss_modules.append(
                AuralossLoss(self.sdstft, 'audio_reals', 'audio_pred', name='mrstft_loss', weight=0.1), # Reconstruction loss
            )

        self.losses = MultiLoss(loss_modules)

    def configure_optimizers(self):
        return optim.Adam([*self.diffae.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        loss_info["audio_reals"] = reals

        if self.diffae.pretransform is not None:
            with torch.no_grad():
                reals = self.diffae.pretransform.encode(reals)

        loss_info["reals"] = reals

        #Encode reals, skipping the pretransform since it was already applied
        latents, encoder_info = self.diffae.encode(reals, return_info=True, skip_pretransform=True)

        loss_info["latents"] = latents
        loss_info.update(encoder_info)

        if self.diffae.decoder is not None:
            latents = self.diffae.decoder(latents)

        # Upsample latents to match diffusion length
        if latents.shape[2] != reals.shape[2]:
            latents = F.interpolate(latents, size=reals.shape[2], mode='nearest')

        loss_info["latents_upsampled"] = latents

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.amp.autocast("cuda"):
            v = self.diffae.diffusion(noised_reals, t, input_concat_cond=latents)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            if self.use_reconstruction_loss:
                pred = noised_reals * alphas - v * sigmas

                loss_info["pred"] = pred

                if self.diffae.pretransform is not None:
                    pred = self.diffae.pretransform.decode(pred)
                    loss_info["audio_pred"] = pred

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': reals.std(),
            'train/latent_std': latents.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffae_ema.update()

    def export_model(self, path, use_safetensors=False):

        model = self.diffae_ema.ema_model

        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class DiffusionAutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionAutoencoderTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        # Remove extra dimension added by WebDataset
        if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
            demo_reals = demo_reals[0]

        encoder_input = demo_reals

        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            latents = module.diffae_ema.ema_model.encode(encoder_input).float()
            fakes = module.diffae_ema.ema_model.decode(latents, steps=self.demo_steps)

        #Interleave reals and fakes
        reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

        # Put the demos together
        reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

        filename = f'recon_{trainer.global_step:08}.wav'
        reals_fakes = reals_fakes.to(torch.float32).div(torch.max(torch.abs(reals_fakes))).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, reals_fakes, self.sample_rate)

        # log_dict[f'recon'] = wandb.Audio(
        #    filename, sample_rate=self.sample_rate, caption=f'Reconstructed')
        # log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
        # log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))
        # log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

        log_audio(
            trainer.logger, "recon", filename,
            sample_rate=self.sample_rate, caption='Reconstructed')
        log_point_cloud(
            trainer.logger, "embeddings_3dpca", pca_point_cloud(latents))
        log_image(
            trainer.logger, "embeddings_spec",
            tokens_spectrogram_image(latents))
        log_image(
            trainer.logger, "recon_melspec_left",
            audio_spectrogram_image(reals_fakes))

        if module.diffae_ema.ema_model.pretransform is not None:
            with torch.no_grad(), torch.amp.autocast("cuda"):
                initial_latents = module.diffae_ema.ema_model.pretransform.encode(encoder_input)
                first_stage_fakes = module.diffae_ema.ema_model.pretransform.decode(initial_latents)
                first_stage_fakes = rearrange(first_stage_fakes, 'b d n -> d (b n)')
                first_stage_fakes = first_stage_fakes.to(torch.float32).mul(32767).to(torch.int16).cpu()
                first_stage_filename = f'first_stage_{trainer.global_step:08}.wav'
                torchaudio.save(first_stage_filename, first_stage_fakes, self.sample_rate)

                log_audio(
                    trainer.logger, "first_stage", first_stage_filename,
                    sample_rate=self.sample_rate, caption='First Stage Reconstructed')
                log_image(
                    trainer.logger, "first_stage_latents",
                    tokens_spectrogram_image(initial_latents))
                log_image(
                    trainer.logger, "first_stage_melspec_left",
                    audio_spectrogram_image(first_stage_fakes))
