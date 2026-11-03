#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/zy/yoda_drift_full/train_full78470}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/train_full_vimeo_drift.py \
  --list-name sep_trainlist_full.txt \
  --epochs 1 \
  --clip-batch 8 \
  --context-chunk 4 \
  --save-every-steps 500 \
  --output-dir "$OUTPUT_DIR" \
  "$@"
