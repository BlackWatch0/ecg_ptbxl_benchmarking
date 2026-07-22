#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_ROOT="${SCENARIO_ROOT:-/content/standard_evaluation_runtime/scenarios}"
TRAINING_ROOT="${TRAINING_ROOT:-/content/drive/MyDrive/ECG/original_baseline_clean_noisy_denoised_v1/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/ECG/original_baseline_clean_noisy_denoised_v1/standardized_evaluation_full_2198}"
CONFIG_ROOT="${CONFIG_ROOT:-/content/standard_evaluation_runtime/evaluation_configs}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-2198}"
if (( $# )); then
  MODELS=("$@")
else
  MODELS=(xresnet1d101 resnet1d_wang fcn_wang inception1d lstm lstm_bidir wavelet_nn)
fi

mkdir -p "$OUTPUT_ROOT" "$CONFIG_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/evaluation_run.log") 2>&1

for required in clean noisy_snr24 noisy_snr12 noisy_snr6 noisy_snr0 noisy_snrm6 \
                denoised_snr24 denoised_snr12 denoised_snr6 denoised_snr0 denoised_snrm6; do
  test -f "$SCENARIO_ROOT/$required.npz" || { echo "Missing scenario $required" >&2; exit 1; }
done
actual_sample_count="$(python -c "import json; print(json.load(open('$SCENARIO_ROOT/scenarios.json'))['sample_count'])")"
test "$actual_sample_count" -eq "$EXPECTED_SAMPLE_COUNT" || {
  echo "Expected $EXPECTED_SAMPLE_COUNT records, found $actual_sample_count" >&2
  exit 1
}
for model in "${MODELS[@]}"; do
  checkpoint="$TRAINING_ROOT/checkpoints/$model/seed_42/checkpoint.pth"
  if [[ "$model" == wavelet_nn ]]; then
    checkpoint="$TRAINING_ROOT/checkpoints/$model/seed_42/best_loss_model.keras"
    feature_count="$(python -c "import json; print(json.load(open('$SCENARIO_ROOT/scenarios.json')).get('wavelet_features', {}).get('feature_count', -1))")"
    test "$feature_count" -eq 864 || {
      echo "Wavelet+NN requires 864 scaled features in every scenario NPZ" >&2
      exit 1
    }
  fi
  test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  test -f "$TRAINING_ROOT/checkpoints/$model/seed_42/thresholds.json" || {
    echo "Missing threshold file for $model" >&2
    exit 1
  }
done

for model in "${MODELS[@]}"; do
  adapter=original_model_factory
  checkpoint="$TRAINING_ROOT/checkpoints/$model/seed_42/checkpoint.pth"
  activation=sigmoid
  call_mode=single
  input_key=ecg
  require_ecg=true
  require_features=false
  feature_shape='[]'
  crop_config=$'  crop_length: 250\n  crop_stride: 125\n  crop_aggregation: max'
  if [[ "$model" == wavelet_nn ]]; then
    adapter=wavelet_keras
    checkpoint="$TRAINING_ROOT/checkpoints/$model/seed_42/best_loss_model.keras"
    activation=identity
    call_mode=feature_only
    input_key=features
    require_ecg=false
    require_features=true
    feature_shape='[864]'
    crop_config=''
  fi
  threshold="$TRAINING_ROOT/checkpoints/$model/seed_42/thresholds.json"
  history="$TRAINING_ROOT/training_logs/$model/seed_42.csv"
  test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  test -f "$threshold" || { echo "Missing threshold file: $threshold" >&2; exit 1; }
  config="$CONFIG_ROOT/$model.yaml"
  cat > "$config" <<YAML
run:
  experiment_name: ${model}_standardized_full_2198
  output_dir: ${OUTPUT_ROOT}/${model}_seed42
  seed: 42
  dataset_split: test
  overwrite: true
  history_file: ${history}
  dataset_name: PTB-XL fold-10 full aligned benchmark
  dataset_version: original-baseline-v1
model:
  name: ${model}
  adapter: ${adapter}
  architecture: ${model}
  checkpoint: ${checkpoint}
  strict_checkpoint: true
  trusted_legacy_checkpoint: false
  num_classes: 5
  input_channels: 12
  activation: ${activation}
  call_mode: ${call_mode}
  input_key: ${input_key}
${crop_config}
data:
  scenarios:
    - {name: clean, condition: clean, path: ${SCENARIO_ROOT}/clean.npz}
    - {name: noisy_snr24, condition: noisy, snr: 24, path: ${SCENARIO_ROOT}/noisy_snr24.npz}
    - {name: noisy_snr12, condition: noisy, snr: 12, path: ${SCENARIO_ROOT}/noisy_snr12.npz}
    - {name: noisy_snr6, condition: noisy, snr: 6, path: ${SCENARIO_ROOT}/noisy_snr6.npz}
    - {name: noisy_snr0, condition: noisy, snr: 0, path: ${SCENARIO_ROOT}/noisy_snr0.npz}
    - {name: noisy_snrm6, condition: noisy, snr: -6, path: ${SCENARIO_ROOT}/noisy_snrm6.npz}
    - {name: denoised_snr24, condition: denoised, snr: 24, path: ${SCENARIO_ROOT}/denoised_snr24.npz}
    - {name: denoised_snr12, condition: denoised, snr: 12, path: ${SCENARIO_ROOT}/denoised_snr12.npz}
    - {name: denoised_snr6, condition: denoised, snr: 6, path: ${SCENARIO_ROOT}/denoised_snr6.npz}
    - {name: denoised_snr0, condition: denoised, snr: 0, path: ${SCENARIO_ROOT}/denoised_snr0.npz}
    - {name: denoised_snrm6, condition: denoised, snr: -6, path: ${SCENARIO_ROOT}/denoised_snrm6.npz}
  batch_size: 64
  num_workers: 0
  input_channels: 12
  ecg_layout: NCT
  require_ecg: ${require_ecg}
  require_features: ${require_features}
  expected_feature_shape: ${feature_shape}
  validate_alignment: true
inference:
  device: ${DEVICE}
  warmup_batches: 1
  timing: both
  dtype: float32
analysis:
  class_names: [NORM, MI, STTC, CD, HYP]
  metrics: true
  calibration: true
  robustness: true
  bootstrap: 1000
  calibration_bins: 10
  threshold:
    mode: load_from_file
    file: ${threshold}
    source_split: validation
output:
  save_predictions: true
  save_logits: true
  save_plots: true
  measure_efficiency: true
YAML
  echo "=== Evaluating $model ==="
  python -u "$PROJECT_ROOT/evaluation/evaluate.py" --config "$config"
done

experiments=()
for model in "${MODELS[@]}"; do experiments+=("${OUTPUT_ROOT}/${model}_seed42"); done
python -u "$PROJECT_ROOT/evaluation/compare_experiments.py" \
  "${experiments[@]}" --output-dir "$OUTPUT_ROOT/comparison" \
  --threshold-strategy load_from_file
echo "Standardized checkpoint evaluations complete: $OUTPUT_ROOT"
