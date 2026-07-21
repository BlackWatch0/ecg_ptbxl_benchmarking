# Standard ECG Evaluation

This package evaluates models independently from training. It never calls `fit`, defaults to strict checkpoint loading, runs PyTorch models under `eval()` and `inference_mode()`, and writes a version-stable artifact tree.

## Data Contract

Each configured scenario is an NPZ with:

- `labels`: binary `[N, K]`
- `sample_id`: unique `[N]`
- `ecg`: optional `[N, C, T]` (or `[N, T, C]` with `ecg_layout: NTC`)
- `features`: optional `[N, ...]`
- `condition`: optional scalar or `[N]`, normally `clean`, `noisy`, or `denoised`
- `snr`: optional scalar or `[N]`

Clean/noisy/denoised scenarios are validated for identical sample IDs, order, and labels. The evaluator does not modify source NPZ files.

## Model Adapters

- `original_model_factory`: xResNet1D101, ResNet1D-Wang, FCN-Wang, Inception1D, LSTM, BiLSTM.
- `cbam_xresnet1d`: ECG-only, CBAM, SE, feature-only, concat/gated EMD late-fusion variants.
- `lightning_checkpoint`: strict repository Lightning checkpoint reconstructions.
- `keras` / `wavelet_keras`: inference-only Keras ECG, feature-only, or two-input fusion checkpoints.
- `factory`: any `module:function` returning a PyTorch module.
- `precomputed_npz`: ID-aligned cached logits/probabilities for model-independent metric migration.

Wavelet/Time-domain or future feature models use `factory`, `call_mode: feature_only` or `late_fusion`, and the same batch dictionary. Register a reusable adapter with `register_model_adapter()` only when a generic factory is insufficient.

Torch adapters require a checkpoint by default. Safe state-dict loading uses `weights_only=True`; legacy Lightning pickle checkpoints require the explicit `trusted_legacy_checkpoint: true` opt-in and must only be used for trusted local artifacts. `allow_uninitialized` is reserved for synthetic smoke tests and must not be used for reported evaluation.

## Run

```bash
python evaluation/evaluate.py \
  --config configs/evaluation/default.yaml \
  --checkpoint checkpoints/best_model.pth \
  --output-dir results/model_seed42 \
  --device cuda --batch-size 256 --num-workers 4 \
  --dataset-split test --seed 42 \
  --threshold-mode load_from_file --threshold-file thresholds.json \
  --evaluate-clean --evaluate-noisy --evaluate-denoised \
  --snr-list 24 12 6 0 -6 --bootstrap 1000 \
  --save-predictions --save-logits --save-plots
```

Threshold search is intentionally absent from the test runner. Generate thresholds on validation data elsewhere and load the resulting artifact.

## Add a Model

1. Prefer a `module:function` factory and configure adapter `factory`.
2. Set `call_mode` to `single`, `feature_only`, or `late_fusion`.
3. Declare `num_classes`, activation, input keys, expected feature shape, and explicit checkpoint key/prefix handling.
4. Add CPU dummy-forward and strict checkpoint tests.
5. Update `model_architecture_summary.md` and `model_architecture_summary.json` as required by `AGENTS.md`.

## Compare and Convert

```bash
python evaluation/compare_experiments.py \
  --results-root results --experiments exp1 exp2 exp3 \
  --output-dir results/comparison

python evaluation/convert_legacy_results.py \
  --input old_ablation_results --output results/converted
```

Legacy conversion leaves unrecoverable fields empty and marks them partial; it never invents predictions, thresholds, IDs, or metrics.
