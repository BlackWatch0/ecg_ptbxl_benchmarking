# PTB-XL CBAM/EMD Ablation Study

This runner trains four 5-class `superdiagnostic` multilabel models with the same folds,
seed, optimizer, OneCycle learning-rate schedule, BCE-with-logits loss, batch size, and
early-stopping rule. The fixed label order is `NORM, MI, STTC, CD, HYP`.

## One Command

Mount Google Drive, then run this single command from the repository root:

```bash
!bash run_ablation_colab.sh
```

The script writes all outputs to `/content/drive/MyDrive/ECG/ablation_results`. It validates
the data before training and invokes the repository's existing download/preparation script if
the required datasets are missing. It runs with `--resume` and writes
`ablation_summary_figures_metrics.zip` to that directory.

## Required Data

`data/ptbxl_clean_no_noise/` must contain the clean PTB-XL metadata, `records100/`, and
`scp_statements.csv`. `data/ptbxl_noisy_mixed_shared/` must contain the noisy WFDB records
and `ptbxl_noisy_mixed_shared_manifest.csv`. The clean EMD CSV is required at
`data/emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv`.

Matched noisy EMD is looked up at the five paths declared in `configs/ablation_cbam_emd.yaml`
and `code/run_ablation_study.py`. If a noisy EMD CSV is absent, the runner emits a warning,
uses the clean EMD feature only as an upper bound, and labels every affected row
`emd_source=clean_original` and `feature_scenario=clean`. It never calls that result matched.

## Manual Invocation

```bash
python code/run_ablation_study.py --config configs/ablation_cbam_emd.yaml --resume
python code/run_ablation_study.py --experiments xresnet1d101_baseline cbam_xresnet1d101 --resume
python code/run_ablation_study.py --smoke-test
```

Use `--seeds 42 123 2026` to run several seeds. The final report includes per-seed rows and
mean/std aggregation. Use `--evaluate-only` or `--skip-training` only after checkpoints exist.

## Results

Each experiment and seed has a checkpoint, epoch-level training history, validation and test
predictions, thresholds, per-class metrics, and scenario integrity checks. The `final_report/`
directory contains comparison, contribution, robustness, and best-model reports. Figures are
written as PNG and PDF without interactive plotting.
