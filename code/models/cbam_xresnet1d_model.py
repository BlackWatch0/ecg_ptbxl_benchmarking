import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset
from pathlib import Path
from functools import partial

from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.train import *
from fastai.torch_core import to_np
from fastai.callbacks.tracker import SaveModelCallback

from models.base_model import ClassificationModel
from models.cbam_xresnet1d import cbam_xresnet1d101
from models.timeseries_utils import aggregate_predictions


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _patched_learner_load(self, name_or_path, with_opt=False, device='cpu', purge=True):
    source = self.path / self.model_dir / '{}.pth'.format(name_or_path)
    state = _load_checkpoint(source, device)
    if set(state.keys()) == {'model', 'opt'}:
        self.model.load_state_dict(state['model'])
        if with_opt:
            self.opt.load_state_dict(state['opt'])
    else:
        self.model.load_state_dict(state)


Learner.load = _patched_learner_load


class PairedTimeseriesDatasetCrops(Dataset):
    def __init__(self, samples, labels, output_size, chunk_length, min_chunk_length,
                 stride=None, random_crop=True):
        self.samples = samples
        self.labels = labels
        self.output_size = output_size
        self.random_crop = random_crop
        self.id_mapping = []
        self.starts = []
        self.ends = []
        for index, sample in enumerate(samples):
            ecg = sample[0]
            length = len(ecg)
            if chunk_length == 0:
                starts, ends = [0], [length]
            else:
                starts = list(range(0, length, chunk_length if stride is None else stride))
                ends = [min(start + chunk_length, length) for start in starts]
            for start, end in zip(starts, ends):
                if end - start >= min_chunk_length:
                    self.id_mapping.append(index)
                    self.starts.append(start)
                    self.ends.append(end)

    def __len__(self):
        return len(self.id_mapping)

    def __getitem__(self, index):
        sample_index = self.id_mapping[index]
        start, end = self.starts[index], self.ends[index]
        if self.random_crop and end - start > self.output_size:
            start += np.random.randint(0, end - start - self.output_size)
        else:
            start += (end - start - self.output_size) // 2
        ecg, emd = self.samples[sample_index]
        return (torch.from_numpy(ecg[start:start + self.output_size].T),
                torch.from_numpy(emd)), torch.from_numpy(self.labels[sample_index])

    def get_id_mapping(self):
        return self.id_mapping


class cbam_xresnet1d_model(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape=None,
                 input_size=2.5, bs=128, wd=1e-2, epochs=50, lr=1e-2,
                 chunkify_train=False, chunkify_valid=True, aggregate_fn='max',
                 loss='binary_cross_entropy', **model_kwargs):
        super().__init__()
        if loss != 'binary_cross_entropy':
            raise ValueError('cbam_xresnet1d_model supports binary_cross_entropy only')
        self.name = name
        self.num_classes = n_classes
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)
        self.input_size = int(input_size * freq)
        self.bs = bs
        self.wd = wd
        self.epochs = epochs
        self.lr = lr
        self.chunkify_train = chunkify_train
        self.chunkify_valid = chunkify_valid
        self.aggregate_fn = aggregate_fn
        self.model_kwargs = model_kwargs

    def _coerce_samples(self, samples):
        result = []
        for sample in samples:
            if self.model_kwargs.get('input_mode', 'late_fusion') == 'ecg_only' and not isinstance(sample, (tuple, list)):
                ecg = np.asarray(sample, dtype=np.float32)
                if ecg.ndim != 2 or ecg.shape[1] != 12:
                    raise ValueError('each ECG sample must have shape [T, 12]')
                result.append((ecg, np.empty((12, 0), dtype=np.float32)))
                continue
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise ValueError('samples must be (ecg [T,12], emd [12,F]) pairs')
            ecg, emd = sample
            ecg, emd = np.asarray(ecg, dtype=np.float32), np.asarray(emd, dtype=np.float32)
            if ecg.ndim != 2 or ecg.shape[1] != 12:
                raise ValueError('each ECG sample must have shape [T, 12]')
            if emd.ndim != 2 or emd.shape[0] != 12:
                raise ValueError('each EMD sample must have shape [12, F]')
            result.append((ecg, emd))
        return result

    def _get_learner(self, train_samples, train_labels, valid_samples, valid_labels):
        emd_features = self.model_kwargs.get('emd_features')
        if emd_features is None and train_samples:
            emd_features = train_samples[0][1].shape[1]
        model_kwargs = dict(self.model_kwargs)
        model_kwargs['emd_features'] = emd_features
        print('CBAM xResNet input_mode={}, use_cbam={}, fusion_type={}, emd_features={}, '
              'feature_hidden_dim={}, feature_embedding_dim={}, fusion_hidden_dim={}'.format(
                  model_kwargs.get('input_mode', 'late_fusion'), model_kwargs.get('use_cbam', True),
                  model_kwargs.get('fusion_type', 'concat'), emd_features,
                  model_kwargs.get('feature_hidden_dim', 256), model_kwargs.get('feature_embedding_dim', 128),
                  model_kwargs.get('fusion_hidden_dim', 256)
              ))
        train_ds = PairedTimeseriesDatasetCrops(
            train_samples, train_labels, self.input_size,
            2 * self.input_size if self.chunkify_train else 0, self.input_size,
            stride=self.input_size // 4, random_crop=True)
        valid_ds = PairedTimeseriesDatasetCrops(
            valid_samples, valid_labels, self.input_size,
            self.input_size if self.chunkify_valid else 0, self.input_size,
            stride=self.input_size // 2, random_crop=False)
        data = DataBunch.create(train_ds, valid_ds, bs=self.bs)
        model = cbam_xresnet1d101(num_classes=self.num_classes, **model_kwargs)
        return Learner(data, model, loss_func=torch.nn.functional.binary_cross_entropy_with_logits,
                       wd=self.wd, path=self.outputfolder)

    def fit(self, X_train, y_train, X_val, y_val):
        X_train, X_val = self._coerce_samples(X_train), self._coerce_samples(X_val)
        y_train = [np.asarray(y, dtype=np.float32) for y in y_train]
        y_val = [np.asarray(y, dtype=np.float32) for y in y_val]
        learner = self._get_learner(X_train, y_train, X_val, y_val)
        inputs, labels = next(iter(learner.data.train_dl))
        ecg, features = inputs
        with torch.no_grad():
            logits = learner.model(ecg, features)
        print('first batch ECG/features/labels/logits:', tuple(ecg.shape), tuple(features.shape),
              tuple(labels.shape), tuple(logits.shape))
        learner.callback_fns.append(partial(SaveModelCallback, monitor='valid_loss', every='improvement',
                                            name='best_valid_loss'))
        learner.fit_one_cycle(self.epochs, self.lr)
        learner.load('best_valid_loss')
        learner.save(self.name, with_opt=True)
        history = pd.DataFrame({
            'epoch': range(len(learner.recorder.val_losses)),
            'valid_loss': [float(loss) for loss in learner.recorder.val_losses]
        })
        history.to_csv(str(self.outputfolder / 'training_history.csv'), index=False)

    def predict(self, X, full_sequence=True):
        probabilities = self._predict(X)
        if not np.isfinite(probabilities).all():
            raise ValueError('Prediction probabilities contain NaN or infinite values')
        if probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError('Fastai get_preds did not return probabilities')
        if probabilities.min() >= 0.5:
            warnings.warn('All probabilities are >= 0.5. Check prediction activation handling.')
        print('probability min/max/mean:', probabilities.min(), probabilities.max(), probabilities.mean())
        return probabilities

    def predict_logits(self, X):
        return self._predict(X, activ=lambda output: output)

    def _predict(self, X, activ=None):
        samples = self._coerce_samples(X)
        labels = [np.ones(self.num_classes, dtype=np.float32) for _ in samples]
        learner = self._get_learner(samples, labels, samples, labels)
        learner.load(self.name)
        predictions, _ = learner.get_preds(activ=activ)
        idmap = np.asarray(learner.data.valid_ds.get_id_mapping())
        aggregate_fn = np.mean if self.aggregate_fn == 'mean' else np.amax
        return aggregate_predictions(to_np(predictions), idmap=idmap, aggregate_fn=aggregate_fn)
