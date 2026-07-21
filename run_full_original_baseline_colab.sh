#!/usr/bin/env bash
# Train original baselines on clean PTB-XL and evaluate clean/noisy/denoised folds.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_ROOT="${ORIGINAL_BASELINE_DRIVE_ROOT:-/content/drive/MyDrive/ECG/original_baseline_clean_noisy_denoised_v1}"
RUNTIME_ROOT="${ORIGINAL_BASELINE_RUNTIME_ROOT:-/content/original_baseline_clean_noisy_denoised_runtime}"
ARCHIVES="$RUNTIME_ROOT/archives"
WORKSPACE="$RUNTIME_ROOT/workspace"
DATA_CONFIG_DIR="$RUNTIME_ROOT/data_config"
RESULTS="$DRIVE_ROOT/results"
LOG_DIR="$DRIVE_ROOT/logs"
LOG_FILE="$LOG_DIR/full_baseline.log"

mkdir -p "$ARCHIVES" "$WORKSPACE" "$DATA_CONFIG_DIR" "$RESULTS" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
on_exit() {
  status=$?
  log "Finished with exit status $status"
  exit "$status"
}
trap on_exit EXIT

declare -A ARCHIVE_NAMES=(
  [clean]='ptb-xl-1.0.3.zip'
  [noisy]='ptbxl_original_database_plus_mixed_WFDB.tar'
  [denoised]='denoised_WFDB.tar'
)
declare -A DRIVE_IDS=(
  [clean]='1SvI2suvuKf4KJ7bikHuGp0PVNAjRJ6Ge'
  [noisy]='1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u'
  [denoised]='1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD'
)

log "Project root: $PROJECT_ROOT"
log "Runtime data root (not persisted): $RUNTIME_ROOT"
log "Drive artifact root: $DRIVE_ROOT"
log "Installing runtime dependencies"
python -m pip install -q gdown wfdb py7zr rarfile pyyaml scikit-learn matplotlib 'pandas==2.2.2' tensorflow

for asset in clean noisy denoised; do
  archive="$ARCHIVES/${ARCHIVE_NAMES[$asset]}"
  if [[ -f "$archive" ]]; then
    log "Reusing $asset archive: $archive ($(du -h "$archive" | cut -f1))"
  else
    log "Downloading $asset archive to runtime storage: $archive"
    gdown --id "${DRIVE_IDS[$asset]}" --output "$archive"
    log "Downloaded $asset archive: $(du -h "$archive" | cut -f1)"
  fi
done

DATA_CONFIG="$DATA_CONFIG_DIR/original_models_benchmark_data.json"
if [[ -f "$DATA_CONFIG" ]]; then
  log "Reusing validated three-condition data config: $DATA_CONFIG"
else
  log "Inspecting and extracting clean/noisy/denoised archives on Drive"
  python "$PROJECT_ROOT/code/prepare_original_models_benchmark_data.py" \
    --archive "$ARCHIVES/${ARCHIVE_NAMES[clean]}" \
    --archive "$ARCHIVES/${ARCHIVE_NAMES[noisy]}" \
    --archive "$ARCHIVES/${ARCHIVE_NAMES[denoised]}" \
    --workspace "$WORKSPACE" \
    --output-dir "$DATA_CONFIG_DIR"
  log "Wrote validated data config: $DATA_CONFIG"
fi

CLEAN_METADATA="$(find "$WORKSPACE" -type f -name 'ptbxl_database.csv' -print -quit)"
if [[ -z "$CLEAN_METADATA" ]]; then
  log "ERROR: Could not locate ptbxl_database.csv below $WORKSPACE"
  exit 1
fi
CLEAN_DATA_ROOT="$(dirname "$CLEAN_METADATA")"
log "Resolved clean training data root: $CLEAN_DATA_ROOT"

log "Starting/resuming 50-epoch baseline benchmark on CUDA"
log "Artifacts: checkpoints, histories, thresholds, integrity checks, predictions, per-class and overall metrics"
python -u "$PROJECT_ROOT/code/run_original_models_benchmark.py" \
  --data-root "$CLEAN_DATA_ROOT" \
  --data-config "$DATA_CONFIG" \
  --output-dir "$RESULTS" \
  --epochs 50 \
  --batch-size 256 \
  --seeds 42 \
  --device cuda \
  --resume

log "Completed baseline benchmark. Results: $RESULTS"
