# Download Link Review

## Active Benchmark Assets

These entries are centralized in `configs/datasets.json` and are used by `run_original_models_benchmark_colab.sh`.

| Key | Purpose | Drive ID | Status |
|---|---|---|---|
| `ptbxl_original` | Clean PTB-XL train/validation/test | `1SvI2suvuKf4KJ7bikHuGp0PVNAjRJ6Ge` | Confirmed current |
| `ptbxl_noisy` | Mixed-noise PTB-XL test domains | `1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u` | Confirmed current |
| `ptbxl_denoised` | Denoised PTB-XL test domains | `1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD` | Confirmed current |
| `wavelet_feature_extraction` | Precomputed wavelet features for original/noisy/denoised domains | `1mGZRk_SJ20miD8DNvK_BjGtQhoJsA60O` | Confirmed current; registered for feature-based workflows |
| `time_domain_feature_extraction` | Beat-level Lead II time-domain/hybrid features for original/noisy/denoised domains | `1wD8Mb216Xd0pjCJhCUrr2nsDhaBaEi-Q` | Confirmed current; Zstandard-compressed tar |

The Wavelet feature archive is `Wavelet feature extraction.tar` (about 330 MB). It contains clean, five mixed-noisy SNR, and five denoised-noisy SNR feature CSV groups. The current original-model benchmark derives Wavelet+NN features directly from ECG and does not yet consume this archive; it is retained in configuration for a later precomputed-feature workflow.

The Time Domain archive contains 11 CSVs: one clean, five mixed-noisy SNR, and five denoised SNR files. Every file uses the same 24-column long-table schema. The matching key is `RecordNumber`, `LeadIndex`, and `BeatIndex`; `RecordNumber` alone is intentionally repeated across beats. All rows use Lead II and include 13 time-domain/hybrid values: `Mean_RR`, `CV_RR`, `pNN50`, `Kurt_RR`, `Skew_P`, `Skew_QRS`, `Skew_ST_T`, `Skew_global`, `RMS_global`, `SD_R_amp`, `SE`, `NTEO`, and `ZCR`. It is Zstandard compressed despite the source archive's generic tar-style name. The Colab-generated detailed inventory is saved under `ECG/time_domain_feature_structure/` in Drive.

## Pending Asset

The previous legacy Drive links have been removed. Clean/noisy/full-ablation scripts now obtain their inputs from the active dataset entries above.

| Asset | Status | Action needed |
|---|---|---|
| EMD late-fusion features | TODO | Provide a replacement archive containing the 11 EMD features per lead required by the EMD workflows. Set its local path through `EMD_ARCHIVE_PATH` until a new active link is registered. |

When the replacement EMD URL is supplied, add it to `configs/datasets.json` and remove the temporary `EMD_ARCHIVE_PATH` requirement.
