# Time-Domain Feature Robustness

`code/run_time_domain_robustness.py` recursively discovers clean, noisy, and denoised CSV, Parquet, or PQ tables below one `--data-root`. Paths must contain a condition label and may contain `snr_5db`, `snr-5`, or `db5`. Rows are matched only by the composite key `RecordNumber`, `LeadIndex`, `BeatIndex`; duplicate condition/SNR/key rows are errors.

The defaults match the registered `time_domain_feature_extraction` archive: `Mean_RR`, `CV_RR`, `pNN50`, `Kurt_RR`, `Skew_P`, `Skew_QRS`, `Skew_ST_T`, `Skew_global`, `RMS_global`, `SD_R_amp`, `SE`, `NTEO`, and `ZCR`. Use `--features` followed by exactly 13 names only for another 13-column schema.

```bash
python code/run_time_domain_robustness.py --data-root data/time_domain_features --output-dir output/time_domain --evaluation-level both --aggregation both --bootstraps 1000 --seed 42
```

The evaluator reports valid/excluded values, adaptive-epsilon NAE, MAE, RMSE, signed and absolute mean/median errors, Pearson/Spearman correlations, raw/scaled cosine similarity, and an unweighted `__macro__` row. It supports beat data and record mean/median aggregation, with confidence intervals from complete-`RecordNumber` cluster bootstraps.

The output contract is `quality_report.csv`, `matching_report.csv`, `feature_metrics.csv`, `macro_overall.csv`, `bootstrap_samples.csv`, `feature_ranking.csv`, `denoising_improvement.csv`, and `sample_errors_top100.csv` (or `sample_errors_all.csv` with `--all-sample-errors`). For every available noisy/denoised comparison, it writes `heatmap_<comparison>_nae.png` and `.pdf`, plus `snr_robustness.png` and `.pdf`.
