"""Names and aliases shared by original benchmark training and orchestration."""

MODEL_NAMES = (
    "xresnet1d101",
    "resnet1d_wang",
    "lstm",
    "lstm_bidir",
    "fcn_wang",
    "inception1d",
)

WAVELET_MODEL_NAME = "wavelet_nn"
SE_MODEL_NAME = "se_xresnet1d101"
BENCHMARK_MODEL_NAMES = MODEL_NAMES + (WAVELET_MODEL_NAME,)

ALIASES = {
    "bidir_lstm": "lstm_bidir",
    "bidirectional_lstm": "lstm_bidir",
    "fastai_xresnet1d101": "xresnet1d101",
    "fastai_resnet1d_wang": "resnet1d_wang",
    "fastai_lstm": "lstm",
    "fastai_lstm_bidir": "lstm_bidir",
    "fastai_fcn_wang": "fcn_wang",
    "fastai_inception1d": "inception1d",
    "wavelet": WAVELET_MODEL_NAME,
    "wavelet+nn": WAVELET_MODEL_NAME,
    "wavelet_nn": WAVELET_MODEL_NAME,
}


def canonical_model_name(name):
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(normalized, normalized)
