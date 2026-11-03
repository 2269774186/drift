#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/amax/Public/zy/dataset}"
DOWNLOAD_SCRIPT="$DATASET_ROOT/download_96f.py"
PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
GPU="${1:-3}"
EXPECTED_FILES=42
EXPECTED_BYTES=298598400
OUTPUT_DIR="$ROOT/outputs/drift_smoke_cuda${GPU}"

cd "$ROOT"

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*"
}

completed_files() {
    find "$DATASET_ROOT" -type f -name '*_96f.yuv' -size "${EXPECTED_BYTES}c" | wc -l
}

log "waiting for dataset downloader"
while pgrep -f '[d]ownload_96f.py' >/dev/null; do
    log "dataset progress: $(completed_files)/$EXPECTED_FILES complete files"
    sleep 60
done

complete="$(completed_files)"
if [ "$complete" -ne "$EXPECTED_FILES" ]; then
    log "download incomplete ($complete/$EXPECTED_FILES); retrying missing files"
    /usr/bin/python3 "$DOWNLOAD_SCRIPT"
    complete="$(completed_files)"
fi

parts="$(find "$DATASET_ROOT" -type f -name '*.part' | wc -l)"
if [ "$complete" -ne "$EXPECTED_FILES" ] || [ "$parts" -ne 0 ]; then
    log "dataset validation failed: complete=$complete/$EXPECTED_FILES parts=$parts"
    exit 2
fi
log "dataset validation passed: $complete files, $EXPECTED_BYTES bytes each"

log "running tests on GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" -m unittest discover -s tests -v
test_status=$?
if [ "$test_status" -ne 0 ]; then
    log "unit tests failed with status $test_status"
    exit "$test_status"
fi

log "starting Drifting smoke training on GPU $GPU"
"$PYTHON" scripts/train_drift_smoke.py \
    --steps 100 \
    --batch-size 4 \
    --frames 8 \
    --latent-size 8 \
    --latent-channels 4 \
    --context-channels 8 4 \
    --hidden-channels 16 \
    --depth 1 \
    --device cuda \
    --output-dir "$OUTPUT_DIR"
train_status=$?

if [ "$train_status" -eq 0 ]; then
    log "experiment completed successfully"
else
    log "experiment failed with status $train_status"
fi
exit "$train_status"
