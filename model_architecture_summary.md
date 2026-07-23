# Model Architecture Summary

Inspected against the current repository on 2026-07-23. Paths and line ranges are
repository-relative. No dataset was loaded and no training was started. Parameter
counts are intentionally not reported because this inventory did not measure and
persist them.

## Verification Record

- Environment: Python 3.13.12, PyTorch 2.12.1+cpu, pytest 9.1.1.
- Existing model forward/factory tests:
  `python -m pytest -q tests/test_ablation_factory.py tests/test_cbam_xresnet1d.py tests/test_se_xresnet1d.py tests/test_original_models_benchmark.py::test_original_model_factory_forward tests/test_original_models_benchmark.py::test_wavelet_alias_uses_feature_factory tests/test_original_models_benchmark.py::test_wavelet_factory_preserves_original_architecture`
- Result: 14 passed, 4 PyTorch `torch.jit.script` deprecation warnings, 7.46 seconds.
- The selected tests use random tensors or a mocked TensorFlow API. They do not load
  PTB-XL and do not run a training loop.
- The repository has no pytest for `wavelet_late_fusion.py` or
  `se_wavelet_late_fusion.py`. A separate `eval()`/`torch.no_grad()` dummy forward
  used ECG `[1,12,1000]` and Wavelet `[1,12,6]`; xResNet, CBAM, and SE ECG-only
  variants and their three Wavelet-fusion counterparts all returned finite `[1,5]`
  logits.

## Current Entry Points and Scope

- The canonical unified orchestrator is `taskmanager.py:1-15`. It delegates to
  `code/task_manager/` and only orchestrates the seven original benchmark models.
- The canonical original-model names and aliases are defined in
  `code/models/original_model_catalog.py:3-33`. `BENCHMARK_MODEL_NAMES` is the six
  raw-waveform names plus `wavelet_nn` (`:3-14`).
- Model construction remains in `code/models/original_model_factory.py:14-63`.
  `code/task_manager/models.py:1-5` imports the catalog, while
  `code/task_manager/runner.py` maps prepare/train/evaluate/report/package to
  the existing Python runners.
- `configs/taskmanager/original_models_benchmark.yaml:21-54` selects exactly
  `xresnet1d101`, `resnet1d_wang`, `lstm`, `lstm_bidir`, `fcn_wang`, `inception1d`,
  and `wavelet_nn`.
- Attention, EMD, Wavelet late-fusion, unified evaluation, and legacy inference
  retain direct Python entries; they are not taskmanager model types.
- Shell launchers are historical files under `scripts/legacy/`. They are not a
  current Bash entry and are not invoked by taskmanager.

## Architecture Overview

| Model or family | ECG input | Feature input | Attention | Fusion/output | Current status |
| --- | --- | --- | --- | --- | --- |
| xResNet1D101 baseline | `[B,12,T]` | none | none | 512-D ECG embedding -> logits | architecture active; EMD runner currently asset-blocked |
| CBAM-xResNet1D101 | `[B,12,T]` | none | CBAM in residual blocks | 512-D ECG embedding -> logits | architecture active; EMD runner currently asset-blocked |
| SE-xResNet1D101 | `[B,12,T]` | none | SE in residual blocks | 512-D ECG embedding -> logits | architecture active; EMD runner currently asset-blocked |
| xResNet/CBAM/SE + EMD | `[B,12,T]` | `[B,12,F]`, normally `F=11` | variant-dependent | 512-D ECG + 128-D EMD; concat or gated -> logits | architecture exists and dummy forward passes; data workflow blocked |
| xResNet + Wavelet | `[B,12,1000]` | `[B,12,6]` | none | 512-D ECG + 128-D Wavelet; concat -> logits | implemented and wired to Wavelet runner |
| CBAM-xResNet + Wavelet | `[B,12,1000]` | `[B,12,6]` | CBAM | 512-D ECG + 128-D Wavelet; concat -> logits | implemented and wired to Wavelet runner |
| SE-xResNet + Wavelet | `[B,12,1000]` | `[B,12,6]` | SE | 512-D ECG + 128-D Wavelet; concat -> logits | architecture/config exist; not wired to current Wavelet runner |
| Six original raw models | `[B,12,250]` | none | model-dependent hooks only | model-specific logits `[B,5]` | taskmanager-supported |
| Original Wavelet+NN | none at classifier boundary | `[B,864]` | none | feature-only Keras probabilities `[B,5]` | taskmanager-supported |
| Legacy FastAI family | cropped ECG | none or legacy EMD pair | model-dependent | model-dependent logits | legacy direct path |
| Lightning reconstructions | `[B,12,T]` | none | model-dependent | checkpoint-compatible logits | inference only |

The fixed superdiagnostic class order is `NORM, MI, STTC, CD, HYP`. PyTorch
training paths produce logits for `BCEWithLogitsLoss`; the original Keras
Wavelet+NN includes sigmoid and produces probabilities.

## 1. Shared xResNet, CBAM, SE, and EMD Architecture

### Definition and flow

- Shared class: `CBAMXResNet1DLateFusion`,
  `code/models/cbam_xresnet1d.py:83-207`.
- Builder: `build_model`, `code/models/cbam_xresnet1d.py:209-234`.
- Experiment declarations: `code/run_ablation_study.py:30-50`.
- Configs: `configs/ablation_cbam_emd.yaml` and `configs/ablation_se.yaml`.

```text
ECG [B,12,T]
-> xResNet stem
-> residual stages (3,4,23,3), expansion 4
-> temporal max pool + temporal mean pool
-> ECG embedding [B,512]

EMD [B,12,F]
-> flatten [B,12F]
-> LayerNorm(12F)
-> Linear(12F,256) -> ReLU -> Dropout(0.3)
-> Linear(256,128) -> ReLU
-> EMD embedding [B,128]

ECG-only: Linear(512,5) -> logits
Feature-only: Linear(128,5) -> logits
Late fusion: concat [B,640]
-> optional sigmoid Linear(640,640) elementwise gate
-> Linear(640,256) -> ReLU -> Dropout(0.4)
-> Linear(256,5) -> logits
```

The ECG backbone construction and max/mean pooling are at
`code/models/cbam_xresnet1d.py:106-118,164-170`. The EMD encoder and fusion head
are at `:120-148,172-195`.

### Attention

- CBAM channel and temporal attention are defined at
  `code/models/cbam_xresnet1d.py:7-34` and applied to each residual branch before
  residual addition at `:56-80`.
- SE is defined at `code/models/cbam_xresnet1d.py:37-53` and is mutually exclusive
  with CBAM. It is applied before residual addition at `:56-80`.
- Defaults are CBAM reduction 16, temporal kernel 7, and SE reduction 16
  (`:58-71,84-89`).

### Training and data status

- The direct ablation loop uses Adam, OneCycleLR, `BCEWithLogitsLoss`, optional
  CUDA AMP, and validation-loss checkpoints:
  `code/run_ablation_study.py:314-388`.
- EMD is standardized using train-fold statistics and the scaler is saved:
  `code/run_ablation_study.py:180-203`.
- The EMD architecture is implemented and covered by current random-tensor tests.
  The data workflow is nevertheless **blocked**: `configs/datasets.json:48-55`
  records `emd_features` as `source_required`, with no active archive. The runner
  requires the clean EMD file before any selected model is run
  (`code/run_ablation_study.py:180-188`). Wavelet or time-domain archives must not
  be substituted for the required 11-feature-per-lead EMD schema.

### Current risks

- `run_ablation_study.py` loads EMD for the whole bundle before model selection,
  so its ECG-only experiments are blocked by the missing EMD asset too.
- `experiment_complete` checks for at least 50 history rows rather than
  `config['epochs']`: `code/run_ablation_study.py:450-458`.
- The class supports feature-only and gated EMD modes, but current experiment
  declarations select ECG-only or concat late fusion only.

## 2. Wavelet Late-Fusion Architectures

Wavelet late fusion is implemented. It is distinct from the original feature-only
Wavelet+NN model.

### Baseline and CBAM Wavelet variants

- Definitions and builder: `code/models/wavelet_late_fusion.py:8-117`.
- `WaveletFeatureEncoder` validates `[B,12,6]`, flattens 72 values, and applies
  `Linear(72,256) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(256,128) ->
  LayerNorm -> GELU`: `:8-25`.
- `WaveletLateFusionXResNet` validates ECG `[B,12,1000]`, pools the xResNet map to
  a 512-D embedding, concatenates a 128-D Wavelet embedding, and applies
  `Linear(640,256) -> GELU -> Dropout(0.2) -> Linear(256,5)`:
  `:28-68`.
- `ECGOnlyXResNet` provides matched baseline and CBAM ECG-only variants:
  `:71-96`.
- The builder permits only no-feature/`fusion_type='none'` or Wavelet
  `[12,6]`/concat combinations: `:99-117`.

### SE Wavelet variants

- Definition and builder: `code/models/se_wavelet_late_fusion.py:8-80`.
- `SEWaveletLateFusionXResNet` uses the same fixed ECG and Wavelet shapes, 512-D
  ECG embedding, 128-D feature embedding, and 640 -> 256 -> 5 concat head:
  `:28-66`.
- With `use_se=True`, `CBAMResBlock` is configured with SE enabled and CBAM
  disabled: `:33-39`.
- Its feature-branch dimensions and dropout are fixed rather than exposed through
  the builder: `:8-20,42-49,69-80`.

### Runner and configuration wiring

- Direct runner: `code/run_wavelet_ablation_study.py`.
- The runner currently imports only `models.wavelet_late_fusion` and declares four
  experiments: baseline, CBAM, baseline+Wavelet, and CBAM+Wavelet
  (`code/run_wavelet_ablation_study.py:20-33`). It builds those models at
  `:351-367`.
- `configs/ablation_cbam_wavelet.yaml:54-58` matches those four names and is the
  default config (`code/run_wavelet_ablation_study.py:51-61`).
- `configs/ablation_se_wavelet.yaml:57-61` declares baseline, SE, baseline+Wavelet,
  and SE+Wavelet. However, the current runner does not import
  `se_wavelet_late_fusion.py`, and its `choices`/`EXPERIMENTS` reject the two SE
  names. The SE architecture and config therefore exist, but that config is not
  currently executable through `run_wavelet_ablation_study.py`.
- The runner fits both ECG and Wavelet standardizers on train folds and reuses them
  for validation/test: `code/run_wavelet_ablation_study.py:235-258`. It enforces
  Wavelet ID alignment and stores features as `[N,12,6]`: `:247-266`.
- Training uses Adam, OneCycleLR, BCE-with-logits, optional CUDA AMP, validation
  loss selection, and configured early stopping: `:291-319`.

## 3. Original Seven-Model Benchmark

### Catalog, factory, and orchestration

- Canonical names/aliases: `code/models/original_model_catalog.py:3-33`.
- PyTorch and Keras factories: `code/models/original_model_factory.py:14-63`.
- Raw model runner: `code/run_original_models_benchmark.py:896-963`.
- Wavelet+NN runner: `code/run_original_models_benchmark.py:801-895`.
- Taskmanager model source: `code/task_manager/models.py:1-5`.

The six raw models receive crop tensors `[B,12,250]` and return logits `[B,5]`.
Crop length and validation/test stride are 250 and 125
(`code/run_original_models_benchmark.py:37-42`). The LSTM models default to LR
0.001; other raw models default to LR 0.01
(`code/models/original_model_factory.py:14-15`).

### Raw model details

- **xResNet1D101:** three-convolution stem, bottleneck stages `(3,4,23,3)`,
  concat max/mean pooling, 128-unit head, dropout 0.5. Definition:
  `code/models/xresnet1d.py:136-184`; factory: `original_model_factory.py:25-28`.
- **Wang ResNet1D:** configured by `resnet1d_wang`,
  `code/models/resnet1d.py:175-189`; factory: `original_model_factory.py:32-33`.
- **LSTM/BiLSTM:** two recurrent layers, hidden size 256, temporal mean/max/state
  pooling, then the shared head. The pooled dimensions are 768 and 1536:
  `code/models/rnn1d.py:17-66`; factory: `original_model_factory.py:34-37`.
- **FCN-Wang:** channels `128 -> 256 -> 128`, kernels `8 -> 5 -> 3`, pooled head:
  `code/models/basic_conv1d.py:125-179`; factory: `original_model_factory.py:38-39`.
- **Inception1D:** six parallel-branch Inception blocks with residual shortcuts
  every three blocks, width 32, scales derived from kernel 40:
  `code/models/inception1d.py:18-104`; factory: `original_model_factory.py:40-41`.

The existing factory tests instantiated all six models and verified finite `[2,5]`
outputs from random `[2,12,250]` tensors.

### Original Wavelet+NN

```text
ECG waveform
-> per-lead db6 level-5 decomposition
-> six coefficient groups * 12 statistics * 12 leads
-> feature vector [B,864]
-> train-fitted StandardScaler
-> Dense(864,128,ReLU)
-> Dropout(0.25)
-> Dense(128,5,sigmoid)
```

- Feature extraction: `code/models/wavelet.py:27-77`.
- Keras factory: `code/models/original_model_factory.py:48-63`.
- Defaults: `code/run_original_models_benchmark.py:40-42` (30 epochs, batch 128),
  overridable through `--wavelet-epochs` and `--wavelet-batch-size` or taskmanager YAML.
- This model is feature-only. It does not negate or replace the separately
  implemented Wavelet+ECG late-fusion models.
- The legacy extractor defaults to `multiprocessing.Pool(18)`:
  `code/models/wavelet.py:68-77`.

## 4. Legacy and Inference-Only Families

- **Legacy FastAI dispatcher:** `code/models/fastai_model.py:159-420`, configured
  by `code/configs/fastai_configs.py` and dispatched by
  `code/experiments/scp_experiment.py`. It includes xResNet, ResNet, Inception,
  FCN/basic/SE CNN, LSTM, and GRU builders.
- **Legacy paired CBAM/EMD wrapper:**
  `code/models/cbam_xresnet1d_model.py`; it adapts paired ECG/EMD inputs to the
  FastAI v1 path. Its data workflow is subject to the same EMD asset block.
- **Lightning checkpoint reconstructions:**
  `code/models/lightning_checkpoint_models.py:18-313`, loaded by
  `code/run_lightning_inference.py`. They reconstruct LeNet, BiLSTM, ResNet,
  Inception, and xResNet checkpoints; no Lightning training module is present.

## Not Implemented as Trainable Fusion Models

- Transformer and CNN-LSTM architectures have no class, builder, config, or
  training entry in the current repository.
- Time-domain robustness code is analysis, not a neural late-fusion branch.
- Wavelet late fusion must not appear in this list: it is implemented in
  `wavelet_late_fusion.py` and `se_wavelet_late_fusion.py`.

## Trace Mapping

```text
canonical unified orchestration
-> taskmanager.py
-> code/task_manager/models.py
-> code/models/original_model_catalog.py
-> code/task_manager/runner.py
-> prepare/run/report Python entries for the seven original models

original benchmark model name
-> original_model_catalog.canonical_model_name
-> original_model_factory.build_original_model or build_wavelet_nn
-> run_original_models_benchmark.run_one or run_wavelet

EMD ablation experiment
-> run_ablation_study.EXPERIMENTS
-> cbam_xresnet1d.build_model
-> CBAMXResNet1DLateFusion.forward

CBAM/baseline Wavelet ablation
-> run_wavelet_ablation_study.EXPERIMENTS
-> wavelet_late_fusion.build_wavelet_ablation_model
-> ECGOnlyXResNet or WaveletLateFusionXResNet

SE Wavelet architecture (not currently runner-wired)
-> configs/ablation_se_wavelet.yaml
-> se_wavelet_late_fusion.build_model
-> SEWaveletLateFusionXResNet
```
