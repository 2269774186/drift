#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_DIR="$ROOT/outputs/real_akiyo_yoda_drift_v4"

if [[ "${1:-}" == "--detach" ]]; then
    LOG_PATH="$OUTPUT_DIR/run_resume.log"
    mkdir -p "$OUTPUT_DIR/artifacts"
    : > "$LOG_PATH"
    nohup bash "$0" > "$LOG_PATH" 2>&1 < /dev/null &
    printf 'detached pid=%s log=%s\n' "$!" "$LOG_PATH"
    exit 0
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR/artifacts"
printf '%s\n' "$$" > "$OUTPUT_DIR/run.pid"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf 'started=%s pid=%s\n' "$(date '+%F %T %z')" "$$"
printf 'python=%s\n' "$PYTHON"
printf 'output_dir=%s\n' "$OUTPUT_DIR"

exec "$PYTHON" scripts/run_real_yoda_drift.py \
    --output-dir "$OUTPUT_DIR" \
    --max-frames 300 \
    --train-frames 200 \
    --train-steps 800 \
    --batch-size 8 \
    --reconstruction-weight 100 \
    --hidden-channels 64 \
    --depth 2 \
    --lr 1e-4 \
    --grad-clip 1 \
    --device cuda:0 \
    --reuse
