#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/data"
OUTPUT_ROOT="${SE_ABLATION_OUTPUT_DIR:-/content/drive/MyDrive/ECG/se_ablation_results}"
FOUR_MODEL_ROOT="${ATTENTION_BASE_RESULTS_DIR:-/content/drive/MyDrive/ECG/ablation_results_full_ptbxl_50_epochs}"
REPORT_ROOT="${OUTPUT_ROOT}/final_report"
LOG="${OUTPUT_ROOT}/training_logs/se_ablation_run.log"

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted at /content/drive/MyDrive" >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is required')
print('GPU: {}'.format(torch.cuda.get_device_name(0)))
PY

python -m pip install -q wfdb pyyaml scikit-learn matplotlib "pandas==2.2.2"
mkdir -p "${OUTPUT_ROOT}/training_logs" "${OUTPUT_ROOT}/errors"

python "${ROOT}/code/colab_data_setup.py" validate --data-root "${DATA_ROOT}"
if [[ ! -d "${FOUR_MODEL_ROOT}/metrics" ]]; then
  echo "Missing completed four-model results: ${FOUR_MODEL_ROOT}" >&2
  exit 1
fi

python -u "${ROOT}/code/run_ablation_study.py" \
  --config "${ROOT}/configs/ablation_se.yaml" \
  --output-dir "${OUTPUT_ROOT}" \
  --experiments se_xresnet1d101 se_xresnet1d101_emd_late_fusion \
  --smoke-test

python -u "${ROOT}/code/run_ablation_study.py" \
  --config "${ROOT}/configs/ablation_se.yaml" \
  --output-dir "${OUTPUT_ROOT}" \
  --experiments se_xresnet1d101 se_xresnet1d101_emd_late_fusion \
  --resume 2>&1 | tee "${LOG}"

python "${ROOT}/code/build_attention_ablation_report.py" \
  --four-model-root "${FOUR_MODEL_ROOT}" \
  --se-root "${OUTPUT_ROOT}" \
  --training-root "${OUTPUT_ROOT}" \
  --output-dir "${REPORT_ROOT}"

python - "${OUTPUT_ROOT}" <<'PY'
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
archive = root / 'se_ablation_summary_figures_metrics.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
    for directory in ('final_report', 'figures', 'metrics', 'config', 'training_logs'):
        for path in (root / directory).rglob('*'):
            if path.is_file():
                output.write(path, path.relative_to(root))
print('Wrote {}'.format(archive))
PY

echo "SE ablation complete: ${OUTPUT_ROOT}"
