# Audio Embedding Inversion

Research code for training embedding-to-audio inversion models, derived from
[Stability AI stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).


## Installation

Use Python 3.10. The package metadata pins the CUDA/PyTorch-era environment used
for this research release, including `torch==2.7.1`, `torchaudio==2.7.1`, and
`pytorch-lightning==2.5.5`.

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Some feature extractors require external checkpoints or packages that are not
redistributed here. Set the corresponding paths in the sample jobs before
running feature extraction.

## Feature Extraction

Training uses precomputed encoder features as conditioning. The generic SLURM
example shows the expected pattern:

```bash
sbatch jobs/sample_extract_features.slurm
```

Edit these variables first:

- `AUDIO_ROOT`: directory containing audio files.
- `FEATURES_ROOT`: directory where `.pt` feature payloads will be written.
- `FILELIST`: optional text file listing the subset to process.
- `ENCODER`: one of `vggish`, `clap`, `beats`, `encodec`, or `convnext`.
- Checkpoint variables such as `CLAP_CKPT`, `CONVNEXT_CKPT`, `VGGISH_CKPT`, and `VGGISH_PCA`.

## Training

The generic training job maps a preset to the matching model config, dataset
metadata loader, and environment variables:

```bash
ENCODER_PRESET=encodec sbatch jobs/sample_train_embedding_inversion.slurm
```

Supported presets:

- `encodec`
- `clap_1sec`
- `clap_5sec`
- `convnext_flattened`
- `convnext_pooled`
- `vggish`

Before submitting, edit the placeholder paths for:

- `AUDIO_ROOT`
- `FEATURES_ROOT`
- `TRAIN_FILELIST`
- `VAL_FILELIST`
- `DEMO_FILELIST`
- `SAVE_DIR`
- `PRETRANSFORM_CKPT`
- `PRETRAINED_CKPT`

The training command itself can be run either through the console entry point or
directly with `train.py`:

```bash
audio-embedding-inversion-train \
  --name embedding_inversion \
  --run_name encodec_example \
  --logger wandb \
  --model-config stable_audio_tools/configs/model_configs/emb2audio/encodec_5_sec_attn_bias.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/examples/train_encodec.json \
  --val-dataset-config stable_audio_tools/configs/dataset_configs/examples/val_encodec.json \
  --demo-dataset-config stable_audio_tools/configs/dataset_configs/examples/demo_encodec.json \
  --batch-size 32 \
  --num-workers 4 \
  --save-dir /path/to/checkpoints \
  --pretransform-ckpt-path /path/to/vae_model.ckpt \
  --pretrained-ckpt-path /path/to/model.safetensors
```

Equivalent direct invocation:

```bash
python -u train.py \
  --name embedding_inversion \
  --run_name encodec_example \
  --logger wandb \
  --model-config stable_audio_tools/configs/model_configs/emb2audio/encodec_5_sec_attn_bias.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/examples/train_encodec.json \
  --val-dataset-config stable_audio_tools/configs/dataset_configs/examples/val_encodec.json \
  --demo-dataset-config stable_audio_tools/configs/dataset_configs/examples/demo_encodec.json \
  --batch-size 32 \
  --num-workers 4 \
  --save-dir /path/to/checkpoints \
  --pretransform-ckpt-path /path/to/vae_model.ckpt \
  --pretrained-ckpt-path /path/to/model.safetensors
```

## Attribution

This repository is a derivative research release, not an official Stability AI
repository. See `ATTRIBUTION.md` and `MODIFIED_FILES.md`.
