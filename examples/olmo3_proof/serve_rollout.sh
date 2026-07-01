#!/usr/bin/env bash
# Start one OLMo3 SGLang rollout server.
#
# Example, per rollout node:
#   MODEL_PATH=/tmp/olmo3_phase2/.../step1100-hf bash serve_rollout.sh 0,1,2,3 30100
#   MODEL_PATH=/tmp/olmo3_phase2/.../step1100-hf bash serve_rollout.sh 4,5,6,7 30101

set -euo pipefail

GPUS="${1:-0,1,2,3}"
PORT="${2:-30100}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/tmp/olmo3_phase2/outputs/phase2_32b_tp8_pp3_seq65536/phase2_32b_tp8_pp3_seq65536/.hf_converted_checkpoints/step1100-hf}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/olmo3_proof}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
TP_SIZE="${TP_SIZE:-4}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-65536}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
OLMO3_USE_SINK="${OLMO3_USE_SINK:-0}"
case "${OLMO3_USE_SINK,,}" in
    1|true|yes|on) OLMO3_SINK_ENABLED=1 ;;
    *) OLMO3_SINK_ENABLED=0 ;;
esac

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

if [[ "$OLMO3_SINK_ENABLED" == "1" ]]; then
    SINK_CACHE_DIR="${OLMO3_SINK_CACHE_DIR:-$RUN_DIR/olmo3_sink_cache}"
    SINK_DEPLOY_DIR="${OLMO3_SGLANG_DEPLOY_DIR:-}"
    PREPARE_ARGS=(
        -m agent_factory_3.models.olmo3.sink_support prepare-sglang
        --model-path "$MODEL_PATH"
        --cache-dir "$SINK_CACHE_DIR"
        --sink-init-value "${OLMO3_SINK_INIT_VALUE:--10.0}"
        --dtype "${OLMO3_SINK_DTYPE:-bfloat16}"
    )
    if [[ -n "$SINK_DEPLOY_DIR" ]]; then
        PREPARE_ARGS+=(--deploy-dir "$SINK_DEPLOY_DIR")
    fi
    if [[ -n "${OLMO3_SINKS_NPZ:-}" ]]; then
        PREPARE_ARGS+=(--sinks-npz "$OLMO3_SINKS_NPZ")
    fi
    if [[ "${OLMO3_SINK_FORCE_CONVERT:-0}" == "1" ]]; then
        PREPARE_ARGS+=(--force-convert)
    fi
    if [[ "${OLMO3_SGLANG_FORCE_DEPLOY:-0}" == "1" ]]; then
        PREPARE_ARGS+=(--force-deploy)
    fi
    echo "[GPU $GPUS] Preparing OLMo3-sink SGLang deploy dir from $MODEL_PATH"
    MODEL_PATH="$("$SGLANG_PYTHON" "${PREPARE_ARGS[@]}" | tail -n 1)"
    echo "[GPU $GPUS] OLMo3-sink deploy dir: $MODEL_PATH"
    if [[ "${OLMO3_SINK_PATCH_SGLANG:-1}" == "1" ]]; then
        bash "$SCRIPT_DIR/apply_olmo3_sink_sglang_patches.sh"
    fi
    export SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"
    export SGLANG_DECODE_NUM_STAGES="${SGLANG_DECODE_NUM_STAGES:-3}"
    export SGLANG_DECODE_BLOCK_N="${SGLANG_DECODE_BLOCK_N:-32}"
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"

echo "[GPU $GPUS] Pre-warming page cache: $MODEL_PATH"
t0=$SECONDS
cat "$MODEL_PATH"/*.safetensors > /dev/null || true
echo "[GPU $GPUS] Cache warm in $((SECONDS - t0))s"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
if [[ -z "$ATTENTION_BACKEND" ]]; then
    if [[ "$OLMO3_SINK_ENABLED" == "1" ]]; then
        ATTENTION_BACKEND="triton"
    else
        ATTENTION_BACKEND="fa3"
    fi
fi

ARGS=(
    --model-path "$MODEL_PATH"
    --tp-size "$TP_SIZE"
    --dtype bfloat16
    --attention-backend "$ATTENTION_BACKEND"
    --kv-cache-dtype fp8_e4m3
    --context-length "$MAX_CONTEXT_TOKENS"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
    --scheduler-recv-interval 16
    --log-level info
    --weight-loader-disable-mmap
    --disable-cuda-graph-padding
)

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
    ARGS+=(--disable-cuda-graph)
fi

if [[ -n "${DFLASH_DRAFT_MODEL_PATH:-}" ]]; then
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
    export SGLANG_DFLASH_DRAFT_RING="${SGLANG_DFLASH_DRAFT_RING:-1}"
    export SGLANG_DFLASH_DRAFT_RING_QUOTA="${SGLANG_DFLASH_DRAFT_RING_QUOTA:-4}"
    export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER="${SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER:-0.125}"
    DFLASH_WINDOW="${DFLASH_WINDOW:-$("$SGLANG_PYTHON" - <<PY
import json
from pathlib import Path
cfg = json.loads((Path("$DFLASH_DRAFT_MODEL_PATH") / "config.json").read_text())
print(cfg.get("sliding_window") or (cfg.get("dflash_config") or {}).get("sliding_window") or 512)
PY
)}"
    ARGS+=(
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "$DFLASH_DRAFT_MODEL_PATH"
        --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE:-8}"
        --speculative-num-draft-tokens "${DFLASH_NUM_DRAFT_TOKENS:-8}"
        --speculative-draft-window-size "$DFLASH_WINDOW"
        --speculative-draft-attention-backend "${DFLASH_DRAFT_ATTENTION_BACKEND:-triton}"
    )
    if [[ -n "${DFLASH_DRAFT_QUANTIZATION:-}" ]]; then
        ARGS+=(--speculative-draft-model-quantization "$DFLASH_DRAFT_QUANTIZATION")
    fi
fi

LOG_FILE="$LOG_DIR/sglang_$(hostname)_port${PORT}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"
exec "$SGLANG_PYTHON" -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
