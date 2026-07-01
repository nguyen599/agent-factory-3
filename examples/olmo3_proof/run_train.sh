#!/usr/bin/env bash
# Launch OLMo3 full-parameter proof RL training on the training node.
#
# Required:
#   MODEL_PATH=/tmp/.../step1100-hf
#   DATASET_PATH=/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv
#   SGLANG_URLS=http://rollout-node-a:30100,http://rollout-node-a:30101,http://rollout-node-b:30100,http://rollout-node-b:30101

set -euo pipefail

GPUS="${1:-0,1,2,3,4,5,6,7}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf}"
DATASET_PATH="${DATASET_PATH:-/tmp/submissions-instructions-runtime/imo_data_1959_2024.csv}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/olmo3_proof}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$SCRIPT_DIR/accelerate_config_olmo3_8gpu.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: DATASET_PATH does not exist: $DATASET_PATH" >&2
    exit 1
fi
if [[ -z "${SGLANG_URLS:-}" ]]; then
    echo "ERROR: SGLANG_URLS must list rollout server URLs" >&2
    exit 1
fi

export MODEL_PATH DATASET_PATH RUN_DIR
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_NO_TQDM=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== OLMo3 proof RL training ==="
echo "  Training GPUs: $GPUS"
echo "  Model:         $MODEL_PATH"
echo "  Dataset:       $DATASET_PATH"
echo "  SGLang URLs:   $SGLANG_URLS"
echo "  Run dir:       $RUN_DIR"
echo "  Verify/meta:   VERIFY_N=${VERIFY_N:-1} META_N=${META_N:-0}"

echo "Pre-warming page cache: $MODEL_PATH"
t0=$SECONDS
cat "$MODEL_PATH"/*.safetensors > /dev/null || true
echo "Cache warm in $((SECONDS - t0))s"

cd "$REPO_DIR"

TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec "$PYTHON_BIN" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    "$SCRIPT_DIR/train.py" 2>&1 | tee "$TRAIN_LOG"
