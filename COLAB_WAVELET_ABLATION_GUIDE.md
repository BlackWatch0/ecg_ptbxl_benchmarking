# Wavelet Late-Fusion Colab

The runner is independent of the EMD experiment code. It uses the mounted archives configured in `configs/ablation_cbam_wavelet.yaml` and never fetches public URLs when a mounted archive is missing.

Run from a Colab checkout of this project after mounting Google Drive:

```bash
!bash run_wavelet_ablation_colab.sh --audit-only
!bash run_wavelet_ablation_colab.sh --smoke-test
!bash run_wavelet_ablation_colab.sh --resume
```

The runner copies archives to `/content/data_archives`, extracts only under `/content/data`, writes Wavelet caches under `/content/cache/wavelet`, and writes results to the configured Drive result directory. Use `--rebuild-cache` after intentionally changing the source Wavelet CSV files or their schema.
