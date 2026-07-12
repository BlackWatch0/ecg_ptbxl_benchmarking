#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/data"
DEFAULT_DOWNLOAD_ROOT="${ROOT}/.downloads"
if [[ -d /content ]]; then
  DEFAULT_DOWNLOAD_ROOT="/content/ecg_downloads"
fi
DOWNLOAD_ROOT="${COLAB_DOWNLOAD_ROOT:-${DEFAULT_DOWNLOAD_ROOT}}"
DEFAULT_OUTPUT_ROOT="${ROOT}/results/ablation_results_full_ptbxl"
if [[ -d /content/drive/MyDrive ]]; then
  DEFAULT_OUTPUT_ROOT="/content/drive/MyDrive/ECG/ablation_results_full_ptbxl"
fi
OUTPUT_ROOT="${FULL_ABLATION_OUTPUT_DIR:-${DEFAULT_OUTPUT_ROOT}}"
CLEAN_PATCH_URL='https://drive.google.com/file/d/1cUF8FSCaGKG4n-QED4NSB4Pb1TSpvEBD/view'
NOISY_PATCH_URL='https://drive.google.com/file/d/14K_jEbRHTnkiP6Qb2qChB7kN6B9UulHE/view'

python -m pip install -q gdown wfdb pyyaml scikit-learn matplotlib
mkdir -p "${DOWNLOAD_ROOT}" "${OUTPUT_ROOT}"

if ! python "${ROOT}/code/colab_data_setup.py" validate --data-root "${DATA_ROOT}"; then
  bash "${ROOT}/colab_run.sh" --prepare
fi

CLEAN_ARCHIVE="${DOWNLOAD_ROOT}/ptbxl_original_noisy_remaining.tar"
NOISY_ARCHIVE="${DOWNLOAD_ROOT}/ptbxl_original_noisy_remaining_plus_mixed_noise.tar"
[[ -f "${CLEAN_ARCHIVE}" ]] || gdown --fuzzy "${CLEAN_PATCH_URL}" --output "${CLEAN_ARCHIVE}"
[[ -f "${NOISY_ARCHIVE}" ]] || gdown --fuzzy "${NOISY_PATCH_URL}" --output "${NOISY_ARCHIVE}"

python "${ROOT}/code/merge_ptbxl_remaining_data.py" merge \
  --data-root "${DATA_ROOT}" \
  --clean-archive "${CLEAN_ARCHIVE}" \
  --noisy-archive "${NOISY_ARCHIVE}" \
  --workspace "${DOWNLOAD_ROOT}"

python -u "${ROOT}/code/run_ablation_study.py" \
  --config "${ROOT}/configs/ablation_cbam_emd.yaml" \
  --output-dir "${OUTPUT_ROOT}" \
  --resume 2>&1 | tee "${OUTPUT_ROOT}/ablation_run.log"

python - "${OUTPUT_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
archive = root / 'ablation_summary_figures_metrics.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
    for directory in ('final_report', 'figures', 'metrics', 'config'):
        for path in (root / directory).rglob('*'):
            if path.is_file():
                output.write(path, path.relative_to(root))
print('Wrote {}'.format(archive))
PY
