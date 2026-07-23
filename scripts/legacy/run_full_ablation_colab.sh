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

python -m pip install -q gdown wfdb pyyaml scikit-learn matplotlib "pandas==2.2.2"
mkdir -p "${DOWNLOAD_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -f "${DATA_ROOT}/ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv" || ! -f "${DATA_ROOT}/ptbxl_noisy_mixed_shared/ptbxl_noisy_mixed_shared_manifest.csv" || ! -f "${DATA_ROOT}/emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv" ]]; then
  bash "${ROOT}/colab_run.sh" --prepare
fi

if [[ ! -f "${DATA_ROOT}/emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv" ]]; then
  echo "EMD feature archive is TODO: set EMD_ARCHIVE_PATH and run bash colab_run.sh --prepare before this ablation." >&2
  exit 1
fi

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
