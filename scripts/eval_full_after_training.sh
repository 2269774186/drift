#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PID="${TRAIN_PID:-3358694}"
TRAIN_DIR="${TRAIN_DIR:-/data1/zy/yoda_drift_full/train_full78470}"
EVAL_DIR="${EVAL_DIR:-/data1/zy/yoda_drift_full/eval_full78470}"
SMOKE_DIR="${SMOKE_DIR:-/data1/zy/yoda_drift_full/eval_smoke78470}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GENERATOR="$TRAIN_DIR/generator.pt"
HISTORY="$TRAIN_DIR/train_history.json"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  command_line="$(tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" 2>/dev/null || true)"
  if [[ "$command_line" != *"train_full_vimeo_drift.py"* ]]; then
    log "PID $TRAIN_PID no longer belongs to the training job"
    exit 1
  fi
  log "waiting for training PID $TRAIN_PID"
  sleep 300
done

log "training process exited"
if [[ ! -s "$GENERATOR" || ! -s "$HISTORY" ]]; then
  log "missing final generator or training history"
  exit 1
fi

"$PYTHON" - "$HISTORY" <<'PY'
import json
import sys
from pathlib import Path

history = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if history.get("status") != "complete":
    raise SystemExit(f"training status is {history.get('status')!r}, not 'complete'")
steps = int(history.get("steps", 0))
selected = int(history.get("selected_clips", 0))
if steps <= 0 or selected != 78470:
    raise SystemExit(
        f"unexpected training scope: steps={steps}, selected_clips={selected}"
    )
print(f"training verified: steps={steps}, selected_clips={selected}")
PY

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "starting one-video full-resolution smoke evaluation"
"$PYTHON" scripts/evaluate_full_benchmark.py \
  --output-dir "$SMOKE_DIR" \
  --generator "$GENERATOR" \
  --training-history "$HISTORY" \
  --max-videos 1 \
  --metric-batch 1 \
  --reuse

log "starting full 42-video benchmark"
"$PYTHON" scripts/evaluate_full_benchmark.py \
  --output-dir "$EVAL_DIR" \
  --generator "$GENERATOR" \
  --training-history "$HISTORY" \
  --metric-batch 1 \
  --reuse

"$PYTHON" - "$EVAL_DIR/benchmark_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "ok":
    raise SystemExit(f"benchmark status is {summary.get('status')!r}")
if int(summary.get("dataset", {}).get("videos", 0)) != 42:
    raise SystemExit("benchmark did not evaluate all 42 videos")
print("full benchmark verified: 42 videos")
PY

log "full benchmark complete: $EVAL_DIR/benchmark_summary.json"
