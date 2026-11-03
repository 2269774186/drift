#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/zy/yoda_drift_full/train_full78470}"
LOG_PATH="${LOG_PATH:-/data1/zy/yoda_drift_full/train_full78470.log}"
PID_PATH="${PID_PATH:-/data1/zy/yoda_drift_full/train_full78470.pid}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")"
nohup env OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/scripts/run_full_vimeo_drift_78470.sh" "$@" \
  >"$LOG_PATH" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_PATH"
printf 'pid=%s log=%s output=%s\n' "$pid" "$LOG_PATH" "$OUTPUT_DIR"
