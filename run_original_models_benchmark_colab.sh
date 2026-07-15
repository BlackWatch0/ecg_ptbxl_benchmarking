#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_ROOT="/content/drive/MyDrive"
OUTPUT_ROOT="${ORIGINAL_MODELS_BENCHMARK_OUTPUT_DIR:-${DRIVE_ROOT}/ECG/original_models_benchmark}"
DOWNLOAD_ROOT="${ORIGINAL_MODELS_BENCHMARK_DOWNLOAD_DIR:-/content/original_models_benchmark_downloads}"
SETUP_ROOT="${ORIGINAL_MODELS_BENCHMARK_DATA_DIR:-/content/original_models_benchmark_data}"
CLEAN_DRIVE_ID="${ORIGINAL_MODELS_CLEAN_DRIVE_ID:-1jWNXSjqUYV0wJOn2BrrmhzOTsVV_cIoM}"
NOISY_DRIVE_ID="${ORIGINAL_MODELS_NOISY_DRIVE_ID:-1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u}"
DENOISED_DRIVE_ID="${ORIGINAL_MODELS_DENOISED_DRIVE_ID:-1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD}"
CLEAN_ARCHIVE="${DOWNLOAD_ROOT}/clean.archive"
NOISY_ARCHIVE="${DOWNLOAD_ROOT}/ptbxl_original_database_plus_mixed_WFDB.tar"
DENOISED_ARCHIVE="${DOWNLOAD_ROOT}/denoised_WFDB.tar"
DATA_CONFIG="${SETUP_ROOT}/normalized/original_models_benchmark_data.json"

if [[ ! -d "${DRIVE_ROOT}" ]]; then
  echo "Google Drive must be mounted at /content/drive before running this script." >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required for the original-model benchmark")
print("GPU: {}".format(torch.cuda.get_device_name(0)))
PY

python -m pip install -q \
  "gdown==5.2.0" \
  "matplotlib==3.9.2" \
  "pandas==2.2.2" \
  "PyYAML==6.0.2" \
  "PyWavelets==1.7.0" \
  "scikit-learn==1.5.2" \
  "wfdb==4.1.2"

python - <<'PY'
try:
    import pywt
    import tensorflow as tf
except ImportError as error:
    raise SystemExit("Wavelet+NN dependency unavailable: {}".format(error))
print("Wavelet+NN runtime: TensorFlow {}, PyWavelets {}".format(tf.__version__, pywt.__version__))
PY

mkdir -p "${DOWNLOAD_ROOT}" "${SETUP_ROOT}/extracted" "${OUTPUT_ROOT}/training_logs"

download_if_absent() {
  local drive_id="$1"
  local destination="$2"
  if [[ -f "${destination}" ]]; then
    echo "Reusing downloaded archive: ${destination}"
    return
  fi
  rm -f "${destination}.part"
  gdown --id "${drive_id}" --output "${destination}.part"
  mv "${destination}.part" "${destination}"
}

download_if_absent "${CLEAN_DRIVE_ID}" "${CLEAN_ARCHIVE}"
python "${ROOT}/code/colab_data_setup.py" prepare \
  --asset clean \
  --archive "${CLEAN_ARCHIVE}" \
  --data-root "${ROOT}/data" \
  --workspace "${DOWNLOAD_ROOT}"

download_if_absent "${NOISY_DRIVE_ID}" "${NOISY_ARCHIVE}"
download_if_absent "${DENOISED_DRIVE_ID}" "${DENOISED_ARCHIVE}"

python "${ROOT}/code/prepare_original_models_benchmark_data.py" \
  --archive "${NOISY_ARCHIVE}" \
  --archive "${DENOISED_ARCHIVE}" \
  --search-root "${ROOT}/data" \
  --workspace "${SETUP_ROOT}/extracted" \
  --output-dir "${SETUP_ROOT}/normalized"

python -u "${ROOT}/code/run_original_models_benchmark.py" \
  --data-config "${DATA_CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/smoke_test" \
  --smoke-test

python -u "${ROOT}/code/run_original_models_benchmark.py" \
  --data-config "${DATA_CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --resume 2>&1 | tee -a "${OUTPUT_ROOT}/training_logs/original_models_benchmark.log"

python "${ROOT}/code/build_original_models_benchmark_report.py" \
  --results-dir "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/final_report"

python - "${OUTPUT_ROOT}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
archive = root / "original_models_benchmark_report.zip"
temporary = archive.with_suffix(".zip.part")
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
    for directory in ("final_report", "metrics", "predictions", "training_logs", "config"):
        source_root = root / directory
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(root))
os.replace(temporary, archive)
print("Wrote {}".format(archive))
PY

echo "Original-model benchmark complete: ${OUTPUT_ROOT}"
