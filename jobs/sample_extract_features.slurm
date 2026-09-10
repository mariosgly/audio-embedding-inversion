#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/extract_features_%A_%a.out
#SBATCH --error=logs/extract_features_%A_%a.err

set -euo pipefail

mkdir -p logs

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENCODER="${ENCODER:-encodec}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/audio}"
FEATURES_ROOT="${FEATURES_ROOT:-/path/to/features}"
FILELIST="${FILELIST:-/path/to/filelist.txt}"
NUM_SHARDS="${NUM_SHARDS:-50}"
DEVICE="${DEVICE:-cuda}"
SCAN_MODE="${SCAN_MODE:-auto}"
MAX_FILES="${MAX_FILES:-0}"

VGGISH_CKPT="${VGGISH_CKPT:-/path/to/vggish_model.ckpt}"
VGGISH_PCA="${VGGISH_PCA:-/path/to/vggish_pca_params.npz}"
CLAP_CKPT="${CLAP_CKPT:-/path/to/clap_checkpoint.pt}"
BEATS_CKPT="${BEATS_CKPT:-/path/to/beats_checkpoint.pt}"
CONVNEXT_CKPT="${CONVNEXT_CKPT:-/path/to/convnext_checkpoint.pth}"

CLAP_MODE="${CLAP_MODE:-chunks}"
CLAP_CHUNK_SEC="${CLAP_CHUNK_SEC:-5.0}"
CLAP_CHUNK_HOP_SEC="${CLAP_CHUNK_HOP_SEC:-1.0}"

cd "${REPO_ROOT}"

python -u feature_extraction/extract_features.py \
  --encoder "${ENCODER}" \
  --audio_root "${AUDIO_ROOT}" \
  --features_root "${FEATURES_ROOT}" \
  --filelist "${FILELIST}" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" \
  --num_shards "${NUM_SHARDS}" \
  --device "${DEVICE}" \
  --scan-mode "${SCAN_MODE}" \
  --max-files "${MAX_FILES}" \
  --write_filelist \
  --filelist_dir "${FEATURES_ROOT}" \
  --filelist_prefix extracted_features \
  --ffmpeg_fallback \
  --vggish_checkpoint_path "${VGGISH_CKPT}" \
  --vggish_pca_params_path "${VGGISH_PCA}" \
  --clap_ckpt_path "${CLAP_CKPT}" \
  --clap_mode "${CLAP_MODE}" \
  --clap_chunk_sec "${CLAP_CHUNK_SEC}" \
  --clap_chunk_hop_sec "${CLAP_CHUNK_HOP_SEC}" \
  --beats_checkpoint_path "${BEATS_CKPT}" \
  --convnext_ckpt_path "${CONVNEXT_CKPT}"
