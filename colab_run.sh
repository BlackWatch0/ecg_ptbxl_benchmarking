#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---train}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/data"
DOWNLOAD_ROOT="${COLAB_DOWNLOAD_ROOT:-/content/ecg_downloads}"
DATASET_CONFIG="${ROOT}/configs/datasets.json"
EMD_ARCHIVE_PATH="${EMD_ARCHIVE_PATH:-}"

if [[ "${MODE}" != "--prepare" && "${MODE}" != "--validate" && "${MODE}" != "--train" ]]; then
  echo "Usage: bash colab_run.sh [--prepare|--validate|--train]"
  exit 2
fi

apt-get -qq update
apt-get -qq install -y p7zip-full
python -m pip install -q gdown wfdb py7zr rarfile
mkdir -p "${DOWNLOAD_ROOT}"

eval "$(python - "${DATASET_CONFIG}" <<'PY'
import json
import shlex
import sys

config = json.load(open(sys.argv[1], encoding='utf-8'))['datasets']
for asset, key in [('clean', 'ptbxl_original'), ('noisy', 'ptbxl_noisy')]:
    item = config[key]
    print('{}_DRIVE_ID={}'.format(asset.upper(), shlex.quote(item['drive_id'])))
    print('{}_ARCHIVE_NAME={}'.format(asset.upper(), shlex.quote(item['archive_name'])))
PY
)"

if [[ "${MODE}" == "--prepare" || "${MODE}" == "--train" ]]; then
  for asset in clean noisy; do
    archive_name_var="${asset^^}_ARCHIVE_NAME"
    drive_id_var="${asset^^}_DRIVE_ID"
    archive="${DOWNLOAD_ROOT}/${!archive_name_var}"
    if [[ ! -f "${archive}" ]]; then
      gdown --id "${!drive_id_var}" --output "${archive}"
    fi
    python "${ROOT}/code/colab_data_setup.py" prepare --asset "${asset}" --archive "${archive}" --data-root "${DATA_ROOT}" --workspace "${DOWNLOAD_ROOT}"
  done
  if [[ -n "${EMD_ARCHIVE_PATH}" ]]; then
    python "${ROOT}/code/colab_data_setup.py" prepare --asset emd --archive "${EMD_ARCHIVE_PATH}" --data-root "${DATA_ROOT}" --workspace "${DOWNLOAD_ROOT}"
  elif [[ "${MODE}" == "--train" ]]; then
    echo "EMD feature archive is TODO: set EMD_ARCHIVE_PATH to a compatible replacement before --train." >&2
    exit 1
  else
    echo "Prepared active clean/noisy assets. EMD feature archive remains TODO; --train requires EMD_ARCHIVE_PATH." >&2
  fi
fi

if [[ "${MODE}" == "--validate" || "${MODE}" == "--train" ]]; then
  python "${ROOT}/code/colab_data_setup.py" validate --data-root "${DATA_ROOT}"
fi

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
