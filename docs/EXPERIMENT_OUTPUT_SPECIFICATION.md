# Experiment Output Specification

Every complete benchmark uses one immutable run directory:

```text
outputs/<experiment_name>_<git_short_hash>_<UTC_timestamp>/
```

Required top-level directories are `config/`, `checkpoints/`, `metrics/`,
`predictions/`, `training_logs/`, `final_report/figures/`, `runtime_logs/`, and
`manifest/`. Feature caches may be retained locally, but all report inputs must
remain in the run root until packaging has completed.

## Required Data

- `config/` stores resolved YAML/JSON, dataset and split manifests, integrity
  checks, label encoder, preprocessing objects, and `artifact_status.json`.
- Every model/seed writes training history, a runtime log, standardized
  `best_model.*` and `last_model.*` checkpoint aliases, and
  `checkpoint_metadata.json`.
- Training history has epoch, train/validation loss, train/validation accuracy,
  learning rate, epoch duration, and best epoch so far. A non-applicable value is
  recorded as empty and explained in checkpoint metadata.
- Metrics, predictions, thresholds, validation metrics, complexity, and every
  configured clean/noisy/denoised integrity report use the common model/seed
  directory layout. Prediction tables use `sample_id`, true labels,
  probabilities, predictions, and per-class thresholds.

## Final Report

`final_report/` contains the benchmark summary tables, Markdown report, and
paired PNG/PDF figures. Per-model loss and training/validation accuracy plots are
required. Learning-rate plots are required when history contains a usable learning
rate. Figure-generation decisions are recorded in
`manifest/figure_generation_status.json`.

## Validation and Recovery

Run `python code/validate_experiment_artifacts.py --input-root <run-root>` after
report generation. It verifies required paths, non-empty files, readable CSV/JSON,
history columns, prediction IDs/counts/probabilities, checkpoint aliases, and
paired figures. It writes expected/actual/missing artifact manifests, checksums,
directory tree, and validation report under `manifest/`.

The package command reruns this validation, writes experiment status, packages the
complete run root, reopens the ZIP, and verifies the required entries. A validation
or archive failure returns non-zero and leaves the run directory intact. Resume the
same resolved configuration and output directory only after repairing the missing
artifact; never mark a partial run as successful.
