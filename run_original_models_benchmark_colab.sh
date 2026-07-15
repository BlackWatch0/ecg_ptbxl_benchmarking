#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_ROOT="/content/drive/MyDrive"
OUTPUT_ROOT="${ORIGINAL_MODELS_BENCHMARK_OUTPUT_DIR:-${DRIVE_ROOT}/ECG/original_models_benchmark}"
DOWNLOAD_ROOT="${ORIGINAL_MODELS_BENCHMARK_DOWNLOAD_DIR:-/content/original_models_benchmark_downloads}"
SETUP_ROOT="${ORIGINAL_MODELS_BENCHMARK_DATA_DIR:-/content/original_models_benchmark_data}"
CLEAN_DRIVE_ID="${ORIGINAL_MODELS_CLEAN_DRIVE_ID:-1SvI2suvuKf4KJ7bikHuGp0PVNAjRJ6Ge}"
NOISY_DRIVE_ID="${ORIGINAL_MODELS_NOISY_DRIVE_ID:-1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u}"
DENOISED_DRIVE_ID="${ORIGINAL_MODELS_DENOISED_DRIVE_ID:-1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD}"
CLEAN_ARCHIVE="${DOWNLOAD_ROOT}/ptb-xl-1.0.3.zip"
CLEAN_DRIVE_CACHE="${ORIGINAL_MODELS_CLEAN_CACHE:-${DRIVE_ROOT}/ECG/datasets/ptbxl_1.0.3_records100.tar}"
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

prepare_clean_ptbxl() {
  if [[ -f "${ROOT}/data/ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv" ]] || \
     [[ -f "${ROOT}/data/ptbxl/ptbxl_database.csv" ]]; then
    local clean_root="${ROOT}/data/ptbxl"
    [[ -f "${ROOT}/data/ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv" ]] && \
      clean_root="${ROOT}/data/ptbxl_clean_no_noise"
    local header_count
    header_count="$(find "${clean_root}/records100" -name '*.hea' 2>/dev/null | wc -l)"
    if [[ "${header_count}" -eq 21799 ]]; then
      echo "Reusing complete clean PTB-XL (${header_count} records)"
      return
    fi
    echo "Ignoring incomplete clean PTB-XL (${header_count}/21799 records)"
  fi
  if [[ -f "${CLEAN_DRIVE_CACHE}" ]]; then
    if [[ -f "${CLEAN_DRIVE_CACHE}.sha256" ]]; then
      (cd "$(dirname "${CLEAN_DRIVE_CACHE}")" && sha256sum -c "$(basename "${CLEAN_DRIVE_CACHE}").sha256")
    fi
    echo "Restoring clean PTB-XL from Drive cache: ${CLEAN_DRIVE_CACHE}"
    tar -xf "${CLEAN_DRIVE_CACHE}" -C "${ROOT}/data"
    [[ "$(find "${ROOT}/data/ptbxl/records100" -name '*.hea' | wc -l)" -eq 21799 ]] || \
      { echo "Drive PTB-XL cache is incomplete" >&2; exit 1; }
    return
  fi
  if download_if_absent "${CLEAN_DRIVE_ID}" "${CLEAN_ARCHIVE}"; then
    prepare_official_ptbxl_zip "${CLEAN_ARCHIVE}"
    return
  fi
  echo "Clean Drive asset unavailable; downloading official PTB-XL 1.0.3 ZIP from PhysioNet"
  rm -f "${CLEAN_ARCHIVE}.part"
  wget -c --show-progress -O "${CLEAN_ARCHIVE}" \
    "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
  prepare_official_ptbxl_zip "${CLEAN_ARCHIVE}"
}

prepare_official_ptbxl_zip() {
  python - "$1" "${ROOT}/data" "${DOWNLOAD_ROOT}/ptbxl_zip_extracted" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

archive, data_root, staging = map(Path, sys.argv[1:])
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)
with zipfile.ZipFile(archive) as source:
    source.extractall(staging)
metadata = list(staging.rglob('ptbxl_database.csv'))
if len(metadata) != 1:
    raise RuntimeError('Expected one ptbxl_database.csv, found {}'.format(metadata))
source = metadata[0].parent
target = data_root / 'ptbxl'
if target.exists():
    shutil.rmtree(target)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(source), str(target))
print('Prepared official PTB-XL: {}'.format(target))
PY
}

prepare_clean_ptbxl

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
