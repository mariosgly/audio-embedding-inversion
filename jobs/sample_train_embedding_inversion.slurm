#!/bin/bash
#SBATCH --job-name=train_embedding_inversion
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_embedding_inversion_%j.out
#SBATCH --error=logs/train_embedding_inversion_%j.err

set -euo pipefail

mkdir -p logs

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENCODER_PRESET="${ENCODER_PRESET:-encodec}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/audio}"
FEATURES_ROOT="${FEATURES_ROOT:-/path/to/features}"
TRAIN_FILELIST="${TRAIN_FILELIST:-/path/to/train.txt}"
VAL_FILELIST="${VAL_FILELIST:-/path/to/val.txt}"
DEMO_FILELIST="${DEMO_FILELIST:-/path/to/demo.txt}"
SAVE_DIR="${SAVE_DIR:-/path/to/checkpoints}"
PRETRANSFORM_CKPT="${PRETRANSFORM_CKPT:-/path/to/vae_model.ckpt}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/model.safetensors}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_NODES="${NUM_NODES:-2}"
STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"
VAL_EVERY="${VAL_EVERY:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-online}"
PROJECT_NAME="${PROJECT_NAME:-embedding_inversion}"

case "${ENCODER_PRESET}" in
  encodec)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/encodec_5_sec_attn_bias.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_encodec.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_encodec.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_encodec.json"
    RUN_NAME="encodec_5sec"
    export ENCODEC_AUDIO_ROOT="${AUDIO_ROOT}"
    export ENCODEC_FEATURES_ROOT="${FEATURES_ROOT}"
    export ENCODEC_NORM="${ENCODEC_NORM:-none}"
    export ENCODEC_SCALE_REDUCTION="${ENCODEC_SCALE_REDUCTION:-mean}"
    ;;
  clap_1sec)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/clap_5_sec_1_sec_embedding.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_clap.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_clap.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_clap.json"
    RUN_NAME="clap_5sec_1sec_embedding"
    export CLAP_AUDIO_ROOT="${AUDIO_ROOT}"
    export CLAP_FEATURES_ROOT="${FEATURES_ROOT}"
    export CLAP_EMB_KEY="${CLAP_EMB_KEY:-embedding_raw}"
    export CLAP_NORM="${CLAP_NORM:-none}"
    export CLAP_EMB_MODE="${CLAP_EMB_MODE:-overlap_all}"
    ;;
  clap_5sec)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/clap_5_sec_5_sec_embedding.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_clap.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_clap.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_clap.json"
    RUN_NAME="clap_5sec_5sec_embedding"
    export CLAP_AUDIO_ROOT="${AUDIO_ROOT}"
    export CLAP_FEATURES_ROOT="${FEATURES_ROOT}"
    export CLAP_EMB_KEY="${CLAP_EMB_KEY:-embedding_raw}"
    export CLAP_NORM="${CLAP_NORM:-none}"
    export CLAP_EMB_MODE="${CLAP_EMB_MODE:-exact_match}"
    ;;
  beats)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/beats_5_sec_flattened_attn_bias.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_beats.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_beats.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_beats.json"
    RUN_NAME="beats_5sec_flattened"
    export BEATS_AUDIO_ROOT="${AUDIO_ROOT}"
    export BEATS_FEATURES_ROOT="${FEATURES_ROOT}"
    ;;
  convnext_flattened)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/convnext_5_sec_flattened_attn_bias.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_convnext.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_convnext.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_convnext.json"
    RUN_NAME="convnext_5sec_flattened"
    export CONVNEXT_AUDIO_ROOT="${AUDIO_ROOT}"
    export CONVNEXT_FEATURES_ROOT="${FEATURES_ROOT}"
    ;;
  convnext_pooled)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/convnext_5_sec_pooled_attn_bias.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_convnext.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_convnext.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_convnext.json"
    RUN_NAME="convnext_5sec_pooled"
    export CONVNEXT_AUDIO_ROOT="${AUDIO_ROOT}"
    export CONVNEXT_FEATURES_ROOT="${FEATURES_ROOT}"
    ;;
  vggish)
    MODEL_CONFIG="stable_audio_tools/configs/model_configs/emb2audio/vggish_5_sec_attn_bias.json"
    TRAIN_CONFIG="stable_audio_tools/configs/dataset_configs/examples/train_vggish.json"
    VAL_CONFIG="stable_audio_tools/configs/dataset_configs/examples/val_vggish.json"
    DEMO_CONFIG="stable_audio_tools/configs/dataset_configs/examples/demo_vggish.json"
    RUN_NAME="vggish_5sec"
    export VGGISH_AUDIO_ROOT="${AUDIO_ROOT}"
    export VGGISH_FEATURES_ROOT="${FEATURES_ROOT}"
    ;;
  *)
    echo "Unknown ENCODER_PRESET=${ENCODER_PRESET}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

TEMPLATE_TRAIN_CONFIG="${TRAIN_CONFIG}"
TEMPLATE_VAL_CONFIG="${VAL_CONFIG}"
TEMPLATE_DEMO_CONFIG="${DEMO_CONFIG}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-generated_configs}"
mkdir -p "${GENERATED_CONFIG_DIR}"

TRAIN_CONFIG="${GENERATED_CONFIG_DIR}/train_${ENCODER_PRESET}.json"
VAL_CONFIG="${GENERATED_CONFIG_DIR}/val_${ENCODER_PRESET}.json"
DEMO_CONFIG="${GENERATED_CONFIG_DIR}/demo_${ENCODER_PRESET}.json"

python - "${TEMPLATE_TRAIN_CONFIG}" "${TRAIN_CONFIG}" "${AUDIO_ROOT}" "${TRAIN_FILELIST}" "train_${ENCODER_PRESET}" <<'PY'
import json
import sys

template_path, out_path, audio_root, filelist, dataset_id = sys.argv[1:]
with open(template_path, "r", encoding="utf-8") as f:
    config = json.load(f)
config["datasets"][0]["id"] = dataset_id
config["datasets"][0]["path"] = audio_root
config["datasets"][0]["filelist"] = filelist
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

python - "${TEMPLATE_VAL_CONFIG}" "${VAL_CONFIG}" "${AUDIO_ROOT}" "${VAL_FILELIST}" "val_${ENCODER_PRESET}" <<'PY'
import json
import sys

template_path, out_path, audio_root, filelist, dataset_id = sys.argv[1:]
with open(template_path, "r", encoding="utf-8") as f:
    config = json.load(f)
config["datasets"][0]["id"] = dataset_id
config["datasets"][0]["path"] = audio_root
config["datasets"][0]["filelist"] = filelist
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

python - "${TEMPLATE_DEMO_CONFIG}" "${DEMO_CONFIG}" "${AUDIO_ROOT}" "${DEMO_FILELIST}" "demo_${ENCODER_PRESET}" <<'PY'
import json
import sys

template_path, out_path, audio_root, filelist, dataset_id = sys.argv[1:]
with open(template_path, "r", encoding="utf-8") as f:
    config = json.load(f)
config["datasets"][0]["id"] = dataset_id
config["datasets"][0]["path"] = audio_root
config["datasets"][0]["filelist"] = filelist
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

export WANDB_MODE="${WANDB_MODE_VALUE}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

srun python -u train.py \
  --name "${PROJECT_NAME}" \
  --run_name "${RUN_NAME}" \
  --logger wandb \
  --num-nodes "${NUM_NODES}" \
  --strategy "${STRATEGY}" \
  --model-config "${MODEL_CONFIG}" \
  --dataset-config "${TRAIN_CONFIG}" \
  --val-dataset-config "${VAL_CONFIG}" \
  --demo-dataset-config "${DEMO_CONFIG}" \
  --val-every "${VAL_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-dir "${SAVE_DIR}" \
  --pretransform-ckpt-path "${PRETRANSFORM_CKPT}" \
  --pretrained-ckpt-path "${PRETRAINED_CKPT}"
