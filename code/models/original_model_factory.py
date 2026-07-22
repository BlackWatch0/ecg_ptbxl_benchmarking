"""Direct PyTorch factory for the original raw-waveform benchmark models."""

from models.basic_conv1d import fcn_wang, weight_init
from models.cbam_xresnet1d import build_model
from models.inception1d import inception1d
from models.resnet1d import resnet1d_wang
from models.rnn1d import RNN1d
from models.xresnet1d import xresnet1d101


MODEL_NAMES = (
    'xresnet1d101',
    'resnet1d_wang',
    'lstm',
    'lstm_bidir',
    'fcn_wang',
    'inception1d',
)

WAVELET_MODEL_NAME = 'wavelet_nn'
SE_MODEL_NAME = 'se_xresnet1d101'
BENCHMARK_MODEL_NAMES = MODEL_NAMES + (WAVELET_MODEL_NAME,)

ALIASES = {
    'bidir_lstm': 'lstm_bidir',
    'bidirectional_lstm': 'lstm_bidir',
    'fastai_xresnet1d101': 'xresnet1d101',
    'fastai_resnet1d_wang': 'resnet1d_wang',
    'fastai_lstm': 'lstm',
    'fastai_lstm_bidir': 'lstm_bidir',
    'fastai_fcn_wang': 'fcn_wang',
    'fastai_inception1d': 'inception1d',
    'wavelet': WAVELET_MODEL_NAME,
    'wavelet+nn': WAVELET_MODEL_NAME,
    'wavelet_nn': WAVELET_MODEL_NAME,
}


def canonical_model_name(name):
    normalized = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    return ALIASES.get(normalized, normalized)


def default_learning_rate(name):
    return 1e-3 if canonical_model_name(name) in ('lstm', 'lstm_bidir') else 1e-2


def build_original_model(name, num_classes=5, input_channels=12):
    """Build one of the six original models with its original benchmark head."""
    name = canonical_model_name(name)
    if name == WAVELET_MODEL_NAME:
        raise NotImplementedError(
            'Wavelet+NN consumes extracted wavelet features; use build_wavelet_nn().'
        )
    common = dict(num_classes=num_classes, input_channels=input_channels,
                  ps_head=0.5, lin_ftrs_head=[128])
    if name == 'xresnet1d101':
        model = xresnet1d101(kernel_size=5, **common)
    elif name == SE_MODEL_NAME:
        model = build_model('xresnet1d101', num_classes, input_channels=input_channels,
                            use_se=True, use_emd=False)
    elif name == 'resnet1d_wang':
        model = resnet1d_wang(kernel_size=5, **common)
    elif name == 'lstm':
        model = RNN1d(lstm=True, bidirectional=False, **common)
    elif name == 'lstm_bidir':
        model = RNN1d(lstm=True, bidirectional=True, **common)
    elif name == 'fcn_wang':
        model = fcn_wang(**common)
    elif name == 'inception1d':
        model = inception1d(kernel_size=40, use_residual=True, **common)
    else:
        raise ValueError('Unknown original model {!r}; expected one of {}'.format(name, MODEL_NAMES))
    model.apply(weight_init)
    return model


def build_wavelet_nn(feature_count, num_classes=5):
    """Build the original Wavelet+NN classifier using modern tf.keras."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            'Wavelet+NN requires TensorFlow; install a Colab-compatible tensorflow package.'
        ) from error
    inputs = tf.keras.Input(shape=(feature_count,))
    hidden = tf.keras.layers.Dense(128, activation='relu')(inputs)
    hidden = tf.keras.layers.Dropout(.25)(hidden)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(hidden)
    model = tf.keras.Model(inputs, outputs, name='Wavelet_NN')
    model.compile(optimizer=tf.keras.optimizers.Adamax(),
                  loss='binary_crossentropy')
    return model
