import torch
import json
import os
import pytorch_lightning as pl
import sys

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def apply_train_filelist_override(dataset_config, filelist):
    if not filelist:
        return dataset_config
    config = json.loads(json.dumps(dataset_config))
    for dataset in config.get("datasets", []):
        dataset["filelist"] = filelist
        dataset["filelist_path"] = filelist
    return config

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Remove demo-dataset-config before prefigure sees it
    demo_dataset_config = None
    if '--demo-dataset-config' in sys.argv:
        idx = sys.argv.index('--demo-dataset-config')
        demo_dataset_config = sys.argv[idx + 1]
        sys.argv.pop(idx)   # remove the flag
        sys.argv.pop(idx)   # remove the value
    
    run_name = None
    if '--run_name' in sys.argv:
        idx = sys.argv.index('--run_name')
        run_name = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    wandb_id = None
    if '--wandb-id' in sys.argv:
        idx = sys.argv.index('--wandb-id')
        wandb_id = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    wandb_resume = None
    if '--wandb-resume' in sys.argv:
        idx = sys.argv.index('--wandb-resume')
        wandb_resume = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    wandb_id_file = None
    if '--wandb-id-file' in sys.argv:
        idx = sys.argv.index('--wandb-id-file')
        wandb_id_file = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    train_filelist_override = None
    if '--train-filelist-override' in sys.argv:
        idx = sys.argv.index('--train-filelist-override')
        train_filelist_override = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
    args = get_all_args()
    seed = args.seed

    is_lightning_resume = bool(args.ckpt_path)
    if not is_lightning_resume and (wandb_id is not None or wandb_resume is not None):
        print(
            "Ignoring --wandb-id/--wandb-resume because --ckpt-path was not provided. "
            "This is a fresh Lightning run, so W&B will create a fresh run id and checkpoint path."
        )
        wandb_id = None
        wandb_resume = None
    
    # print('running main function')

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    # print('seeding everything')
    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    dataset_config = apply_train_filelist_override(dataset_config, train_filelist_override)

    training_config = model_config.get("training", {})
    matmul_precision = training_config.get("float32_matmul_precision", None)

    if isinstance(matmul_precision, str):
        normalized_matmul_precision = matmul_precision.strip().lower()
        if normalized_matmul_precision not in {"off", "none", "false", ""}:
            valid_precisions = {"highest", "high", "medium"}
            if normalized_matmul_precision not in valid_precisions:
                raise ValueError(
                    f"Invalid training.float32_matmul_precision={matmul_precision!r}. "
                    f"Expected one of {sorted(valid_precisions)} or null/off."
                )
            torch.set_float32_matmul_precision(normalized_matmul_precision)
            print(f"Set torch float32 matmul precision to '{normalized_matmul_precision}' from config")
    elif isinstance(matmul_precision, bool):
        if matmul_precision:
            torch.set_float32_matmul_precision("high")
            print("Set torch float32 matmul precision to 'high' from config")
    elif matmul_precision is not None:
        raise ValueError(
            f"Invalid training.float32_matmul_precision={matmul_precision!r}. "
            "Expected a string, boolean, or null."
        )

    # print('will create the training dataloader')
    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        # print('will create the validation dataloader')
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False #False
        )
        demo_dl = None
   
    demo_dl = None
    if demo_dataset_config is not None:
        with open(demo_dataset_config) as f:
            demo_dataset_config_json = json.load(f)

        print('I have a test demo so I am creating a dataloader for that')
        demo_bs = getattr(args, "demo_batch_size", None) or min(8, args.batch_size)

        demo_dl = create_dataloader_from_config(
            demo_dataset_config_json,
            batch_size=demo_bs,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=True,
        )

    model = create_model_from_config(model_config)
    
    # total = 0
    # trainable = 0

    # for name, param in model.named_parameters():
    #     total += param.numel()
    #     if param.requires_grad:
    #         trainable += param.numel()
    #         print("TRAIN:", name)
    #     else:
    #         print("FROZEN:", name)

    # print(f"Total params: {total}")
    # print(f"Trainable params: {trainable}")

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"
        logger = pl.loggers.WandbLogger(project=args.name, name=run_name, id=wandb_id, resume=wandb_resume)
        logger.watch(training_wrapper)

        if wandb_id_file and isinstance(logger.experiment.id, str):
            wandb_id_file_dir = os.path.dirname(wandb_id_file)
            if wandb_id_file_dir:
                os.makedirs(wandb_id_file_dir, exist_ok=True)
            with open(wandb_id_file, "w") as f:
                f.write(logger.experiment.id + "\n")
            print(f"Wrote W&B run id to {wandb_id_file}: {logger.experiment.id}")

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        dirpath=checkpoint_dir,
        save_top_k=3,
        monitor="val/avg_loss",
        mode="min",
        filename="epoch={epoch}-step={step}-val_avg_loss={val/avg_loss:.6f}",
        auto_insert_metric_name=False,
        save_last=True,
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    # if args.val_dataset_config:
    #     demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    # else:
    #     demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
        # Prefer demo_dl if provided, otherwise fall back to val_dl, otherwise train_dl
    if demo_dl is not None:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=demo_dl)
    elif args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args).copy()
    for runtime_key in ("ckpt_path", "pretrained_ckpt_path", "pretransform_ckpt_path"):
        args_dict.pop(runtime_key, None)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
