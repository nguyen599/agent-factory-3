#!/usr/bin/env bash
# Apply the author-provided OLMo3-sink SGLang patches to a writable Python env.
#
# Usage:
#   SGLANG_PYTHON=python bash examples/olmo3_proof/apply_olmo3_sink_sglang_patches.sh
#   bash examples/olmo3_proof/apply_olmo3_sink_sglang_patches.sh /path/to/venv --verify-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV="${1:-${SGLANG_VENV:-}}"
MODE="${2:-apply}"

if [[ -z "$VENV" ]]; then
    PY="${SGLANG_PYTHON:-python}"
    VENV="$("$PY" - <<'PY'
import sys
print(sys.prefix)
PY
)"
fi

find_patch_root() {
    if [[ -n "${OLMO3_SINK_SGLANG_PATCH_ROOT:-}" ]]; then
        echo "$OLMO3_SINK_SGLANG_PATCH_ROOT"
        return
    fi
    for candidate in \
        "$REPO_DIR/../patch-olmo3-sink-infer" \
        "$REPO_DIR/../../patch-olmo3-sink-infer" \
        "$PWD/patch-olmo3-sink-infer"; do
        if [[ -f "$candidate/deploy_kaggle/apply_dflash_patches.sh" ]]; then
            echo "$(cd "$candidate" && pwd)"
            return
        fi
    done
    return 1
}

PATCH_ROOT="$(find_patch_root)" || {
    echo "ERROR: patch-olmo3-sink-infer not found; set OLMO3_SINK_SGLANG_PATCH_ROOT" >&2
    exit 1
}

echo "[olmo3-sink-patch] venv=$VENV"
echo "[olmo3-sink-patch] patch_root=$PATCH_ROOT"

if [[ "${OLMO3_SINK_SGLANG_PATCH_SET:-sink}" == "all" ]]; then
    bash "$PATCH_ROOT/kaggle_deploy/final/serve/apply_all_patches.sh" "$VENV" "${MODE/apply/}"
else
    bash "$PATCH_ROOT/deploy_kaggle/apply_dflash_patches.sh" "$VENV" "${MODE/apply/}"
    if [[ "$MODE" == "--verify-only" ]]; then
        python "$PATCH_ROOT/deploy_kaggle/patch_decode_tune.py" "$VENV" --verify-only
        python "$PATCH_ROOT/deploy_kaggle/patch_gqa_packed_extend.py" "$VENV" --verify-only
    else
        python "$PATCH_ROOT/deploy_kaggle/patch_decode_tune.py" "$VENV"
        python "$PATCH_ROOT/deploy_kaggle/patch_gqa_packed_extend.py" "$VENV"
        if [[ "${OLMO3_SINK_APPLY_W4A8_PATCH:-0}" == "1" ]]; then
            bash "$PATCH_ROOT/deploy/w4a8/apply_w4a8_patch.sh" "$VENV"
        fi
    fi
fi

echo "[olmo3-sink-patch] done"
