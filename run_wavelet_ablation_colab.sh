#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python - <<'PY'
import importlib.util
import torch
required = ('numpy', 'pandas', 'sklearn', 'wfdb', 'yaml')
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing dependencies: ' + ', '.join(missing))
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

test -f /content/drive/MyDrive/ECG_Project/data/ptbxl/ptbxl_records100.tar.gz
test -f /content/drive/MyDrive/ECG_Project/data/wavelet/Wavelet_Features_Complete.zip
test -f /content/drive/MyDrive/ECG_Project/data/noisy/ptbxl_original_database_plus_mixed_WFDB.tar
test -f /content/drive/MyDrive/ECG_Project/data/denoised/denoised_WFDB.tar

python code/run_wavelet_ablation_study.py --config configs/ablation_cbam_wavelet.yaml "$@"
