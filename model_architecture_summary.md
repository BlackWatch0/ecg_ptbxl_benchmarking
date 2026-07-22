# Model Architecture Summary

Generated from a static repository inspection on 2026-07-21. Paths below are repository-relative. No dataset was loaded and no training was started. Dummy forward tests were attempted conceptually but not run because the local analysis environment has no `torch` (`ModuleNotFoundError: No module named 'torch'`); no large dependency was installed.

## Overview

| Model name | Backbone | Attention | Feature branch | Fusion | ECG embedding | Feature embedding | Parameter count | Code status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| xResNet1D101 baseline | xResNet-101 | none | none | none | 512 in current ablation implementation | - | unavailable locally | active ablation variant |
| CBAM-xResNet1D101 | xResNet-101 | CBAM | none | none | 512 | - | unavailable locally | active ablation variant |
| SE-xResNet1D101 | xResNet-101 | SE | none | none | 512 | - | unavailable locally | active ablation variant |
| xResNet1D101 + EMD | xResNet-101 | none | EMD MLP | concat or gated | 512 | 128 | unavailable locally | active concat variant |
| CBAM-xResNet1D101 + EMD | xResNet-101 | CBAM | EMD MLP | concat or gated | 512 | 128 | unavailable locally | active concat variant |
| SE-xResNet1D101 + EMD | xResNet-101 | SE | EMD MLP | concat or gated | 512 | 128 | unavailable locally | active concat variant |
| Original xResNet1D101 | xResNet-101 | implementation-dependent hooks only | none | none | classifier head input from backbone pool | - | unavailable locally | active original benchmark |
| Wang ResNet1D | 3-stage Wang ResNet | none | none | none | pooled convolutional features | - | unavailable locally | active original benchmark |
| LSTM / BiLSTM | 2-layer recurrent | temporal max/mean/state pooling | none | none | 768 / 1536 | - | unavailable locally | active original benchmark |
| FCN-Wang | 3-layer FCN | none | none | none | pooled convolutional features | - | unavailable locally | active original benchmark |
| Inception1D | InceptionTime-style | residual shortcuts, not attention | none | none | pooled inception features | - | unavailable locally | active original benchmark |
| Wavelet+NN | handcrafted db6 features + Keras MLP | none | Wavelet statistics | feature-only | - | 864 input / 128 hidden | unavailable locally | active original benchmark |
| Legacy FastAI configurable family | xResNet/ResNet/Inception/FCN/RNN builders | model-dependent | none | none | model-dependent | - | unavailable locally | active legacy dispatcher |
| Lightning checkpoint reconstructions | LeNet/LSTM/ResNet/Inception/xResNet | model-dependent | none | none | model-dependent | - | unavailable locally | inference-only reconstruction |

All classification paths are multi-label ECG classification where their active configurations use `BCEWithLogitsLoss` or Keras binary cross-entropy. The active CBAM/EMD and original benchmark tasks use five superdiagnostic classes: `NORM`, `MI`, `STTC`, `CD`, `HYP`.

## Shared Input Conventions

- Active ablation ECG input: `[batch, 12, length]`; 12 leads are enforced by `CBAMXResNet1DLateFusion._validate_ecg` in `code/models/cbam_xresnet1d.py:150-154`.
- Active EMD input: `[batch, 12, features]`; current common EMD schema contains 11 features/lead, thus the default tensor is `[B, 12, 11]`. Evidence: `code/utils/emd_features.py:8-14`, `code/models/cbam_xresnet1d.py:156-160`.
- Legacy/ablation ECG sampling rate is 100 Hz. The legacy CBAM configuration uses 10-second, 1,000-sample inputs: `code/configs/cbam_configs.py:11-35`.
- Original benchmark raw models receive crop tensors `[B, 12, 250]`; validation/test crops use stride 125: `code/run_original_models_benchmark.py:36-41,317-321`.
- Original benchmark Wavelet+NN consumes `[B, 864]` (`12 leads * 6 db6 coefficient groups * 12 statistics`): `code/run_original_models_benchmark.py:38-41`.

## 1. xResNet1D101 Baseline, CBAM, and SE Variants

**Basic information**

- Shared class: `CBAMXResNet1DLateFusion`; definition: `code/models/cbam_xresnet1d.py:83-203`.
- Builder: `build_model('xresnet1d101', ...)`, `code/models/cbam_xresnet1d.py:206-224`.
- Current training entry: `code/run_ablation_study.py:529-634`.
- Variants are declared in `code/run_ablation_study.py:33-40`: `xresnet1d101_baseline`, `cbam_xresnet1d101`, and `se_xresnet1d101`.
- Configs: `configs/ablation_cbam_emd.yaml`, `configs/ablation_se.yaml`.
- Static construction status: constructible by code path; local forward was not executed because `torch` is unavailable.

**Input/output**

```text
ECG input: [B, 12, T]
Feature input: none for ECG-only variants
Output logits: [B, 5] in active ablations
```

**Backbone and classifier data flow**

```text
ECG
-> XResNet1d stem: Conv1d/BN/ReLU stack (32, 32, 64) + max pool
-> residual stages with xResNet-101 layers (3, 4, 23, 3), expansion 4
-> final feature map
-> temporal max pool and temporal mean pool
-> concat ECG embedding [B, 512]
-> Linear(512, 5) for ECG-only mode
```

The underlying stem, residual construction, adaptive pooling and classifier implementation are in `code/models/xresnet1d.py:97-191`. The fusion wrapper deliberately removes the original xResNet classifier (`list(backbone.children())[:-1]`) and computes max+mean pooling itself: `code/models/cbam_xresnet1d.py:106-116,164-170`.

**Attention**

- CBAM is inserted inside every `CBAMResBlock`, after the residual convolution path and before residual addition: `code/models/cbam_xresnet1d.py:56-80`.
- Channel attention: temporal average and maximum pooling, shared 1x1 Conv MLP, ReLU, sigmoid; default reduction 16: `:7-20`.
- Temporal attention: channel average and maximum maps, concatenation, `Conv1d(2,1,kernel=7)`, sigmoid: `:23-34`.
- SE is mutually exclusive with CBAM. It uses adaptive global temporal pooling and 1x1 Conv/ReLU/Conv/sigmoid scale, default reduction 16: `:37-53,63-71`.

**Training**

- `BCEWithLogitsLoss`, Adam, OneCycleLR, AMP when CUDA available: `code/run_ablation_study.py:333-380`.
- YAML defaults: 50 epochs, batch 128, LR .01, weight decay .01, seed 42: `configs/ablation_cbam_emd.yaml:1-22`.
- Best checkpoint is selected by validation BCE; last checkpoint is written every epoch: `code/run_ablation_study.py:397-406`.

**Potential issues**

- `se_reduction` is recorded in metadata but not passed into `build_model` from `run_ablation_study.py`; the model therefore uses its default 16 regardless of a different YAML value. Evidence: `code/run_ablation_study.py:529-545`, `code/models/cbam_xresnet1d.py:206-224`.
- `early_stopping_patience`, `optimizer`, `loss`, `monitor`, and `save_best_only` in ablation YAML are declarative; the loop hard-codes Adam/BCE/best-valid-loss behavior and runs all epochs. Evidence: `run_ablation_study.py:333-407`.
- All ablation selections load and align clean EMD before knowing whether a selected model uses EMD. A waveform-only baseline therefore still requires EMD assets: `run_ablation_study.py:193-207`.

## 2. EMD Late-Fusion Variants

**Basic information**

- Shared implementation: `CBAMXResNet1DLateFusion`, `code/models/cbam_xresnet1d.py:83-203`.
- Active variants: `xresnet1d101_emd_late_fusion`, `cbam_xresnet1d101_emd_late_fusion`, `se_xresnet1d101_emd_late_fusion`: `code/run_ablation_study.py:37-39`.
- Active configuration selects concat fusion; gated fusion is supported by the class but is not selected by that experiment dictionary.

**Feature branch and fusion**

```text
EMD [B, 12, 11]
-> Flatten [B, 132]
-> LayerNorm(132)
-> Linear(132, 256)
-> ReLU
-> Dropout(0.3)
-> Linear(256, 128)
-> ReLU
-> EMD embedding [B, 128]

ECG [B, 12, T]
-> xResNet feature map
-> temporal max + mean concat
-> ECG embedding [B, 512]

concat: [B, 640]
-> Linear(640, 256) -> ReLU -> Dropout(0.4)
-> Linear(256, 5) logits
```

Evidence: `code/models/cbam_xresnet1d.py:120-148,172-195`. In gated mode the same 640-D vector is multiplied elementwise by `sigmoid(Linear(640,640))` before the fusion MLP: `:139-142,184-185`.

The EMD standardizer is fit on training folds only and saved as `emd_scaler.npz`: `code/run_ablation_study.py:204-217`; `code/utils/emd_features.py:116-128`.

**Potential issues**

- There is LayerNorm on flattened EMD features but no explicit branch norm matching ECG embedding scale before concat; branch dominance is possible and requires empirical diagnostics.
- The 12-lead and `12 * emd_features` dimensions are hard-coded in the encoder and validator: `code/models/cbam_xresnet1d.py:123-124,156-160`.
- `feature_dropout` is configurable in the class but not exposed by `build_model`'s named arguments; it can only arrive through `**kwargs`.
- The class outputs logits, correctly leaving sigmoid to prediction/evaluation code. Do not apply sigmoid before `BCEWithLogitsLoss`.

## 3. Original Benchmark Raw-Waveform Models

**Factory and training entry**

- Factory: `code/models/original_model_factory.py:10-70`.
- Active trainer: `code/run_original_models_benchmark.py:744-879,897-963`.
- Models: `xresnet1d101`, `resnet1d_wang`, `lstm`, `lstm_bidir`, `fcn_wang`, `inception1d`.
- Shared input/output: ECG crop `[B,12,250]` -> logits `[B,5]`; BCE-with-logits, Adam, OneCycleLR, mixed precision. The LSTM LR is `.001`; other raw models use `.01`: `code/models/original_model_factory.py:42-70`.
- Current full Bash entry: `run_full_original_baseline_colab.sh`; it trains on clean folds 1-8, selects thresholds on fold 9, and evaluates fold 10 clean/noisy/denoised scenarios.

### 3.1 Original xResNet1D101

- Class/factory: `XResNet1d` / `xresnet1d101`, `code/models/xresnet1d.py:136-191`.
- Flow: 3-convolution stem -> max pool -> bottleneck residual stages `(3,4,23,3)` -> adaptive max+average pool -> flattened head -> logits.
- Original factory supplies a 128-unit classifier head and `.5` head dropout: `code/models/original_model_factory.py:46-70`.

### 3.2 Wang ResNet1D

- Definition: `code/models/resnet1d.py:175-189`; factory mapping: `code/models/original_model_factory.py:51-53`.
- Flow: three residual stages, each one block, 128 channels, no stem pooling -> shared adaptive max+average head -> logits.

### 3.3 LSTM and Bidirectional LSTM

- Definition: `code/models/rnn1d.py:17-75`; factory: `code/models/original_model_factory.py:54-57`.
- Flow: transpose `[B,C,T]` to `[B,T,C]` -> two-layer LSTM, hidden 256 -> temporal mean, temporal max, and terminal state concatenation -> MLP head -> logits.
- Embedding dimensions are 768 unidirectional and 1536 bidirectional: `code/models/rnn1d.py:51-64`.
- No transformer architecture is defined or configured in this repository.

### 3.4 FCN-Wang

- Definition and factory: `code/models/basic_conv1d.py:125-188`.
- Flow: Conv1d/BN/activation stages with channels `128 -> 256 -> 128` and kernels `8 -> 5 -> 3` -> adaptive max+average pooling -> configurable head -> logits.
- No attention in this configured FCN variant.

### 3.5 Inception1D

- Definition: `code/models/inception1d.py:18-104`.
- Flow: six Inception blocks; each block applies a 1x1 bottleneck then parallel temporal convolutions at three scales plus max-pool/1x1 branch -> concat/BN/ReLU. Residual shortcuts occur every three blocks -> concat-pool head -> logits.
- Default block width is 32; kernel scales are derived from 40: `code/models/inception1d.py:45-86`.

**Potential issues**

- Random crop selection excludes the final valid crop start because it calls `randint(0, maximum - 1)`: `code/run_original_models_benchmark.py:70-73`.
- The runner now writes raw-model history and `last_checkpoint.pth` atomically after every epoch: `code/run_original_models_benchmark.py:330-339,433-457`. This is a persistence property, not an architecture change.

## 4. Wavelet + NN

**Basic information**

- Feature extraction: `code/models/wavelet.py:27-77`.
- Keras model factory: `code/models/original_model_factory.py:73-88`.
- Active benchmark path: `code/run_original_models_benchmark.py:558-747`.

```text
ECG [B, 12, T]
-> db6 level-5 decomposition per lead
-> entropy/statistics/crossings for six coefficient groups
-> Wavelet vector [B, 864]
-> StandardScaler fit on clean training folds only
-> Dense(864, 128, ReLU)
-> Dropout(0.25)
-> Dense(128, 5, sigmoid)
```

- Keras uses Adamax and binary cross-entropy, 30 epochs, batch 128: `code/run_original_models_benchmark.py:909-919`; `code/models/original_model_factory.py:73-88`.
- Unlike PyTorch models, the Keras output is probabilities because sigmoid is inside the final Dense layer.
- Last and best Keras models plus CSV history are checkpointed each epoch: `code/run_original_models_benchmark.py:599-628`.

**Potential issues**

- Legacy `WaveletModel.get_ecg_features` defaults to `multiprocessing.Pool(18)`, which is resource-rigid: `code/models/wavelet.py:68-77`.
- This is feature-only classification, not Wavelet late fusion. No Wavelet+ECG fusion model is implemented.

## 5. Legacy FastAI Configurable Family

- Definition/dispatcher: `code/models/fastai_model.py:159-420`; training dispatch: `code/experiments/scp_experiment.py:120-176`.
- Name prefixes route to xResNet (including deep/deeper), ResNet, Inception, FCN/basic/SE CNN, LSTM and GRU builders: `code/models/fastai_model.py:328-402`.
- Input crops are transformed from time-major records to channels-first model tensors; output is model logits, trained with FastAI BCE-with-logits: `code/models/fastai_model.py:210-286,302-312`.
- Config variants are in `code/configs/fastai_configs.py:1-141`; legacy reproduction entry is `code/reproduce_results.py:17-44`.

**Potential issues**

- `SCP_Experiment.perform` handles unknown `modeltype` with `assert(True); break`, silently stopping rather than raising: `code/experiments/scp_experiment.py:149-155`.
- Fine-tuning mismatch handling appears to pass `model.__dict__` as constructor kwargs: `code/experiments/scp_experiment.py:285-291`.

## 6. Legacy CBAM-EMD FastAI Wrapper

- Dataset wrapper: `PairedTimeseriesDatasetCrops`, `code/models/cbam_xresnet1d_model.py:41-80`.
- It yields ECG crop `[B,12,T]` and EMD `[B,12,F]`; model construction uses `cbam_xresnet1d101`, BCE-with-logits and FastAI one-cycle fitting: `code/models/cbam_xresnet1d_model.py:125-171`.
- Configs: `code/configs/cbam_configs.py:11-98`; active launcher selects late fusion only: `code/run_cbam_emd_experiment.py:8-15`.
- Defined `ecg_only` and `feature_only` legacy configs are not selected by a repository launcher.

## 7. Lightning Checkpoint Reconstructions

- Definitions: `code/models/lightning_checkpoint_models.py:18-313`.
- Supported checkpoint architectures: LeNet, bidirectional LSTM, bottleneck ResNet, Inception-v3-like 1D model, xResNet.
- Loader validates key/tensor compatibility: `load_checkpoint_model`, `:290-313`.
- Inference entry: `code/run_lightning_inference.py:89-220`.
- Status: inference-only reconstruction; no Lightning training module or active training entry was found.

## Feature Types Not Implemented as Trainable Fusion Branches

- **Time-domain features:** `code/time_domain_robustness/` and `code/run_time_domain_robustness*.py` are tabular robustness analysis pipelines, not a neural feature branch or late-fusion classifier.
- **Wavelet feature robustness:** `code/wavelet_feature_snr_robustness.py` and its notebook analyze precomputed features; they do not define a trainable fusion architecture.
- **Transformer/CNN-LSTM:** no actual Transformer or CNN-LSTM class, builder, config, or training entry was found.

## Key Architectural Differences

- CBAM-xResNet differs from xResNet only by channel then temporal attention within residual blocks; SE is an alternative channel recalibration path.
- Late fusion occurs after ECG max+mean pooling, before the final classifier; it is embedding-level fusion, not logits-level fusion.
- EMD uses a flattened, globally normalized 132-D vector for the default 11-feature schema. Wavelet+NN is feature-only and uses 864 handcrafted inputs. Time-domain has no neural branch.
- Concat and gated fusion share dimensions; gated fusion adds a 640-to-640 sigmoid gate before the same fusion MLP.
- Inception1D uses parallel temporal convolution branches and periodic residual shortcuts; xResNet uses sequential bottleneck residual stages.
- Recurrent models process time as sequence and use aggregate temporal/state features; CNN/xResNet/Inception models use convolutional feature maps and pooling.

## Code Trace Mapping

```text
ablation experiment name
-> run_ablation_study.EXPERIMENTS
-> models.cbam_xresnet1d.build_model
-> CBAMXResNet1DLateFusion.forward
-> run_ablation_study.train_model

original benchmark model name
-> models.original_model_factory.BENCHMARK_MODEL_NAMES
-> build_original_model / build_wavelet_nn
-> model.forward / Keras model.fit
-> run_original_models_benchmark.run_one / run_wavelet

legacy FastAI configuration
-> configs.fastai_configs or configs.cbam_configs
-> SCP_Experiment.perform
-> fastai_model or cbam_xresnet1d_model
-> FastAI Learner.fit_one_cycle
```

## Unused or Suspected Legacy Items

- `code/models/your_model.py` and `code/configs/your_configs.py` are explicit placeholders; `fit` and `predict` are empty.
- Lightning architecture files reconstruct checkpoints but have no training definition.
- Legacy CBAM `ecg_only` and `feature_only` config variants are defined but lack an active launcher.
- `code/test_evaluate_exp0.py` runs a legacy experiment at import time and is not a normal test.
- `code/run_inference.py` and `code/recover_cbam_emd_predictions.py` contain hard-coded legacy relative paths.

## Dummy Instantiation Record

```text
Attempted environment check: .venv/bin/python -c 'import torch'
Result: ModuleNotFoundError: No module named 'torch'
Successful model instantiations: 0
Failed/unavailable dummy forwards: all PyTorch architectures (environment dependency unavailable)
TensorFlow/Keras instantiation: not attempted; TensorFlow is not installed in the local analysis environment.
```
