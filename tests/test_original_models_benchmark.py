import random
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.original_model_factory import (MODEL_NAMES, build_original_model,
                                           build_wavelet_nn,
                                           canonical_model_name,
                                           default_learning_rate)
from run_original_models_benchmark import (CLASS_NAMES, WAVELET_BATCH_SIZE,
                                           WAVELET_EPOCHS,
                                           WAVELET_FEATURE_COUNT, CropDataset,
                                           build_loader,
                                           cache_wavelet_features,
                                           load_manifest_scenario,
                                           predict_crops, train_model,
                                           train_wavelet_classifier)


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_original_model_factory_forward(name):
    model = build_original_model(name)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 12, 250))
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
    assert default_learning_rate(name) == (1e-3 if 'lstm' in name else 1e-2)


def test_wavelet_alias_uses_feature_factory():
    assert canonical_model_name('Wavelet+NN') == 'wavelet_nn'
    with pytest.raises(NotImplementedError, match='build_wavelet_nn'):
        build_original_model('wavelet_nn')


def test_wavelet_factory_preserves_original_architecture(monkeypatch):
    operations = []

    class Tensor:
        def __init__(self, width):
            self.width = width

    class Dense:
        def __init__(self, units, activation):
            operations.append(('dense', units, activation))
            self.units = units

        def __call__(self, value):
            return Tensor(self.units)

    class Dropout:
        def __init__(self, rate):
            operations.append(('dropout', rate))

        def __call__(self, value):
            return value

    class Model:
        def __init__(self, inputs, outputs, name):
            self.output_width = outputs.width

        def compile(self, optimizer, loss):
            operations.append(('compile', type(optimizer).__name__, loss))

    fake_tf = SimpleNamespace(keras=SimpleNamespace(
        Input=lambda shape: Tensor(shape[0]),
        layers=SimpleNamespace(Dense=Dense, Dropout=Dropout),
        Model=Model,
        optimizers=SimpleNamespace(Adamax=type('Adamax', (), {}))))
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)
    model = build_wavelet_nn(WAVELET_FEATURE_COUNT)
    assert model.output_width == 5
    assert operations == [
        ('dense', 128, 'relu'), ('dropout', .25), ('dense', 5, 'sigmoid'),
        ('compile', 'Adamax', 'binary_crossentropy')]


def test_wavelet_feature_cache_is_id_aligned_and_reused(tmp_path):
    calls = []
    waveforms = np.zeros((2, 1000, 12), dtype=np.float32)
    ids = np.array([10, 20])

    def extract(values):
        calls.append(len(values))
        return np.arange(2 * WAVELET_FEATURE_COUNT, dtype=np.float32).reshape(2, -1)

    first = cache_wavelet_features(waveforms, ids, 'clean', tmp_path, extract)
    second = cache_wavelet_features(waveforms, ids, 'clean', tmp_path,
                                    lambda values: pytest.fail('cache was not reused'))
    assert calls == [2]
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match='misaligned'):
        cache_wavelet_features(waveforms, ids[::-1], 'clean', tmp_path, extract)


def test_wavelet_training_uses_train_only_scaler_and_original_schedule(tmp_path, monkeypatch):
    fit_calls, loaded = [], []

    class Callback:
        pass

    class ModelCheckpoint:
        def __init__(self, filepath, **kwargs):
            self.filepath = Path(filepath)

    class FakeModel:
        trainable_weights = []

        def fit(self, features, labels, **kwargs):
            fit_calls.append((features.copy(), kwargs))
            for callback in kwargs['callbacks']:
                if isinstance(callback, ModelCheckpoint):
                    callback.filepath.touch()
                elif hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(0, {'loss': .4, 'val_loss': .3})

        def evaluate(self, features, labels, verbose=0):
            return .3

    model = FakeModel()
    fake_tf = SimpleNamespace(keras=SimpleNamespace(
        callbacks=SimpleNamespace(Callback=Callback, ModelCheckpoint=ModelCheckpoint),
        models=SimpleNamespace(load_model=lambda path: loaded.append(Path(path)) or model),
        utils=SimpleNamespace(set_random_seed=lambda seed: None)))
    monkeypatch.setattr('run_original_models_benchmark._tensorflow', lambda: fake_tf)
    monkeypatch.setattr('run_original_models_benchmark.build_wavelet_nn', lambda size: model)
    train = np.vstack([np.zeros(WAVELET_FEATURE_COUNT), np.full(WAVELET_FEATURE_COUNT, 2)])
    valid = np.full((1, WAVELET_FEATURE_COUNT), 100)
    labels = np.zeros((2, 5), dtype=np.float32)
    checkpoint = tmp_path / 'checkpoints'
    checkpoint.mkdir()
    history = tmp_path / 'history.csv'
    _, scaler, _, _, _, _ = train_wavelet_classifier(
        train, labels, valid, labels[:1], checkpoint, history, 42)
    assert np.allclose(scaler.mean_, 1)
    assert fit_calls[0][1]['epochs'] == WAVELET_EPOCHS
    assert fit_calls[0][1]['batch_size'] == WAVELET_BATCH_SIZE
    assert fit_calls[0][1]['initial_epoch'] == 0

    train_wavelet_classifier(
        train, labels, valid, labels[:1], checkpoint, history, 42, resume=True)
    assert fit_calls[1][1]['initial_epoch'] == 1
    assert checkpoint / 'last_model.keras' in loaded


def test_crop_dataset_and_max_probability_aggregation():
    waveforms = np.zeros((2, 1000, 12), dtype=np.float32)
    labels = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
    dataset = CropDataset(waveforms, labels, training=False)
    assert len(dataset) == 14
    assert [start for record, start in dataset.items if record == 0] == [0, 125, 250, 375, 500, 625, 750]

    class CropScore(torch.nn.Module):
        def forward(self, value):
            score = value[:, 0].mean(dim=1, keepdim=True)
            return score.repeat(1, 5)

    waveforms[0, 625:875, 0] = 2.0
    loader = build_loader({'ecg': waveforms, 'labels': labels}, False, 4)
    probabilities, found_labels, _ = predict_crops(CropScore(), loader, torch.device('cpu'), 2)
    assert np.allclose(probabilities[0], torch.sigmoid(torch.tensor(2.0)).item())
    assert np.array_equal(found_labels, labels)


def test_manifest_loading_aligns_by_ecg_id_and_rejects_missing(tmp_path):
    from sklearn.preprocessing import StandardScaler

    for ecg_id in (10, 20):
        np.save(tmp_path / '{}.npy'.format(ecg_id), np.full((250, 12), ecg_id, dtype=np.float32))
    manifest = pd.DataFrame({
        'ecg_id': [20, 10], 'snr_db': [6, 6],
        'path': ['20.npy', '10.npy'],
    })
    path = tmp_path / 'manifest.csv'
    manifest.to_csv(path, index=False)
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))
    waveforms, integrity = load_manifest_scenario(path, tmp_path, 6, np.array([10, 20]), scaler)
    assert waveforms[0, 0, 0] < waveforms[1, 0, 0]
    assert integrity['record_order_matches_test_ecg_ids'] is True
    with pytest.raises(ValueError, match='missing ecg_id'):
        load_manifest_scenario(path, tmp_path, 6, np.array([10, 30]), scaler)


def test_training_checkpoint_resumes_at_next_epoch(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.output = torch.nn.Linear(12, 5)

        def forward(self, value):
            return self.output(self.pool(value).squeeze(-1))

    random.seed(1)
    waveforms = np.random.RandomState(1).normal(size=(4, 250, 12)).astype(np.float32)
    labels = np.zeros((4, 5), dtype=np.float32)
    labels[np.arange(4), np.arange(4)] = 1
    split = {'ecg': waveforms, 'labels': labels}
    train = build_loader(split, True, 2)
    valid = build_loader(split, False, 2)
    best, last, history = tmp_path / 'best.pth', tmp_path / 'last.pth', tmp_path / 'history.csv'
    base = {'model_name': 'fcn_wang', 'learning_rate': 1e-2,
            'weight_decay': 1e-2, 'epochs': 1, 'mixed_precision': False}
    train_model(TinyModel(), train, valid, base, torch.device('cpu'),
                best, last, history)
    resumed = dict(base, epochs=2)
    train_model(TinyModel(), train, valid, resumed, torch.device('cpu'),
                best, last, history, resume=True)
    assert pd.read_csv(history).epoch.tolist() == [1, 2]
    assert torch.load(last, map_location='cpu', weights_only=False)['epoch'] == 2
