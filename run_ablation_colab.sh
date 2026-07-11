#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_OUTPUT="${ABLATION_OUTPUT_DIR:-/content/drive/MyDrive/ECG/ablation_results}"
RUN_LOG="${DRIVE_OUTPUT}/ablation_run.log"

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted. Mount it first with: from google.colab import drive; drive.mount('/content/drive')"
  exit 2
fi

python -m pip install -q wfdb pyyaml scikit-learn matplotlib

if ! python "${ROOT}/code/colab_data_setup.py" validate --data-root "${ROOT}/data"; then
  echo "Prepared data is incomplete; downloading and preparing the configured datasets."
  bash "${ROOT}/colab_run.sh" --prepare
fi
mkdir -p "${DRIVE_OUTPUT}"
RUN_ARGS=(
  --config "${ROOT}/configs/ablation_cbam_emd.yaml"
  --output-dir "${DRIVE_OUTPUT}"
  --resume
)
if [[ "${ABLATION_SMOKE_ONLY:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke-test)
fi

if ! python -u "${ROOT}/code/run_ablation_study.py" \
  "${RUN_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"; then
  echo "Ablation runner failed. Full traceback: ${RUN_LOG}"
  exit 1
fi

if [[ "${ABLATION_SMOKE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

ARCHIVE="${DRIVE_OUTPUT}/ablation_summary_figures_metrics.zip"
python - "${DRIVE_OUTPUT}" "${ARCHIVE}" <<'PY'
import sys
import zipfile
from pathlib import Path

root, archive = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
    for directory in ('final_report', 'figures', 'metrics', 'config'):
        for path in (root / directory).rglob('*'):
            if path.is_file():
                output.write(path, path.relative_to(root))
print('Wrote {}'.format(archive))
PY
