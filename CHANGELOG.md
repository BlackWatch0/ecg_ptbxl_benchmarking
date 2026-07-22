# Changelog

## Unreleased

### Added

- Central `configs/datasets.json` registry for active PTB-XL benchmark assets.
- Wavelet feature extraction archive registration for clean, noisy, and denoised scenarios.
- Time-domain feature extraction archive registration and beat-level schema documentation.
- `docs/DOWNLOAD_LINKS_REVIEW.md` listing active and legacy Drive assets requiring review.

### Changed

- The original-model Colab benchmark loads active PTB-XL asset IDs and archive names from the dataset registry.
- Legacy clean/noisy/full-ablation download links were replaced with active dataset registry entries.

### Removed

- Expired legacy Drive URLs and the generated download-link audit file.

### Pending

- A replacement EMD late-fusion feature archive is required before EMD workflows can run again.

## 2026-07-16

### Added

- Original-model three-domain benchmark pipeline for seven paper-reproduction models.
- SE-xResNet1D attention ablation pipeline and six-model reporting workflow.

### Fixed

- PTB-XL benchmark fold-10 validation ignores non-test records present in full noisy and denoised archives.
- Colab benchmark supports user-supplied PTB-XL ZIP assets and Drive cache restoration.
