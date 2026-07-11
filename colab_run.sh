#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---train}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/data"
DOWNLOAD_ROOT="${COLAB_DOWNLOAD_ROOT:-/content/ecg_downloads}"

declare -A URLS
URLS[noisy]='https://drive.google.com/file/d/145xQ_s23q3Ee0HVSxAs_Dq4q-uqMdxk5/view?usp=sharing'
URLS[clean]='https://drive.google.com/file/d/1jWNXSjqUYV0wJOn2BrrmhzOTsVV_cIoM/view'
URLS[emd]='https://drive.google.com/file/d/1mnHzcZcWo2rQVYRmjojarTXhp6ouRYus/view?usp=drive_link'

if [[ "${MODE}" != "--prepare" && "${MODE}" != "--validate" && "${MODE}" != "--train" ]]; then
  echo "Usage: bash colab_run.sh [--prepare|--validate|--train]"
  exit 2
fi

apt-get -qq update
apt-get -qq install -y p7zip-full
python -m pip install -q gdown wfdb py7zr rarfile
mkdir -p "${DOWNLOAD_ROOT}"

if [[ "${MODE}" == "--prepare" || "${MODE}" == "--train" ]]; then
  for asset in clean noisy emd; do
    archive="${DOWNLOAD_ROOT}/${asset}.archive"
    if [[ ! -f "${archive}" ]]; then
      gdown --fuzzy "${URLS[$asset]}" --output "${archive}"
    fi
    python "${ROOT}/code/colab_data_setup.py" prepare --asset "${asset}" --archive "${archive}" --data-root "${DATA_ROOT}" --workspace "${DOWNLOAD_ROOT}"
  done
fi

python "${ROOT}/code/colab_data_setup.py" validate --data-root "${DATA_ROOT}"

if [[ "${MODE}" == "--train" ]]; then
  python -m pip install -q fastai==1.0.61
  cd "${ROOT}/code"

  echo "=== Verifying 5-class labels ==="
  python verify_5class.py

  OLD_OUTPUT="${ROOT}/output/exp_emd_late_fusion_superdiagnostic"
  if [[ -d "${OLD_OUTPUT}" ]]; then
    echo "=== Removing previous output directory ==="
    rm -rf "${OLD_OUTPUT}"
  fi

  echo "=== Training CBAM-xResNet1D + EMD late fusion ==="
  python run_cbam_emd_experiment.py

  echo "=== Diagnosing validation metrics ==="
  python diagnose_cbam_emd.py

  echo "=== Evaluating all SNR scenarios ==="
  python evaluate_cbam_emd_snr.py
fi
