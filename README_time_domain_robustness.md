# Time-Domain Feature Robustness

`code/run_time_domain_robustness.py` recursively discovers clean, noisy, and denoised CSV, Parquet, or PQ tables below one `--data-root`. Paths must contain a condition label and may contain `snr_5db`, `snr-5`, or `db5`. Rows are matched only by the composite key `RecordNumber`, `LeadIndex`, `BeatIndex`; duplicate condition/SNR/key rows are errors.

The defaults match the registered `time_domain_feature_extraction` archive: `Mean_RR`, `CV_RR`, `pNN50`, `Kurt_RR`, `Skew_P`, `Skew_QRS`, `Skew_ST_T`, `Skew_global`, `RMS_global`, `SD_R_amp`, `SE`, `NTEO`, and `ZCR`. Use `--features` followed by exactly 13 names only for another 13-column schema.

```bash
python code/run_time_domain_robustness.py --data-root data/time_domain_features --output-dir output/time_domain --evaluation-level both --aggregation both --bootstraps 1000 --seed 42
```

The evaluator reports valid/excluded values, adaptive-epsilon NAE, MAE, RMSE, signed and absolute mean/median errors, Pearson/Spearman correlations, raw/scaled cosine similarity, and an unweighted `__macro__` row. It supports beat data and record mean/median aggregation, with confidence intervals from complete-`RecordNumber` cluster bootstraps.

The output contract is `quality_report.csv`, `matching_report.csv`, `feature_metrics.csv`, `macro_overall.csv`, `bootstrap_samples.csv`, `feature_ranking.csv`, `denoising_improvement.csv`, and `sample_errors_top100.csv` (or `sample_errors_all.csv` with `--all-sample-errors`). For every available noisy/denoised comparison, it writes `heatmap_<comparison>_nae.png` and `.pdf`, plus `snr_robustness.png` and `.pdf`.

## V2 metric contract

`code/run_time_domain_robustness_v2.py` is a separate evaluator. It does not alter the legacy/debug output contract above.

```bash
python code/run_time_domain_robustness_v2.py --data-root data/time_domain_features --output-dir output/time_domain_v2 --bootstrap-iterations 1000 --seed 42
```

V2 uses exactly 13 features. Feature NMAE is MAE divided by that feature's clean `p95 - p05`; a zero or invalid percentile span falls back first to the clean median absolute value and then to one. `v2_feature_metrics.csv` records the scale, percentiles, and fallback method. The `__macro_13d__` NMAE is strict: it is reported only when all 13 feature NMAEs are available. Raw and scaled cosines are row-wise cosine similarities of the full 13-dimensional paired feature vectors, not an average of scalar feature cosines. Scaled vectors divide each coordinate by its clean scale.

The exact V2 artifacts are `v2_input_manifest.csv`, `v2_quality_report.csv`, `v2_overlap_audit.csv`, `v2_feature_metrics.csv`, `v2_macro_metrics.csv`, `v2_bootstrap_samples.csv`, `v2_heatmap_noisy_nmae.png/.pdf`, `v2_heatmap_denoised_nmae.png/.pdf`, and `v2_macro_nmae_by_snr.png/.pdf`. The manifest stores root-relative POSIX paths and SHA-256 digests. The overlap audit reports composite-key coverage before metrics are calculated. Bootstrap draws resample complete `RecordNumber` clusters and use record-level sufficient statistics for strict macro NMAE and the two 13D cosines.

V2 limitations: it measures feature agreement, not clinical equivalence or diagnostic performance; composite-key overlap cannot establish that signals were generated from the same acquisition; p95-p05 normalization is sensitive to a small or nonrepresentative clean reference; and confidence intervals quantify sampling variation under record-cluster resampling, not preprocessing, feature-extraction, or noise-generation uncertainty.
