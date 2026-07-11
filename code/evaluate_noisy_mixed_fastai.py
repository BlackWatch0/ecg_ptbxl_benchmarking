import sys
sys.path.insert(0, '.')

import ast
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from fastai.basic_train import Learner

_original_learner_load = Learner.load


def _patched_load(self, name_or_path, with_opt=False, device='cpu'):
    if isinstance(name_or_path, str) and not name_or_path.endswith('.pth'):
        source = self.path / self.model_dir / f'{name_or_path}.pth'
    else:
        source = Path(name_or_path)
    state = torch.load(str(source), map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        if with_opt:
            self.opt.load_state_dict(state['opt'])
        self.model.load_state_dict(state['model'])
    else:
        self.model.load_state_dict(state)


Learner.load = _patched_load

from models.fastai_model import fastai_model
from utils import utils


SCRIPT_VERSION = '0.1.0'
DATA_ROOT = Path('../data')
OUTPUT_ROOT = Path('../output')
NOISY_ROOT = DATA_ROOT / 'ptbxl_noisy_mixed_shared'
CLEAN_ROOT = DATA_ROOT / 'ptbxl_clean_no_noise'
EXPERIMENT_ROOT = OUTPUT_ROOT / 'exp0'
RESULT_ROOT = OUTPUT_ROOT / 'noisy_mixed_shared' / 'fastai_xresnet1d101'


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as file:
        return CompatibleUnpickler(file).load()


def load_noisy_test_set(manifest, snr, test_ids, scaler):
    rows = manifest[(manifest.snr_target_db == snr) & manifest.ecg_id.isin(test_ids)]
    paths = dict(zip(rows.ecg_id, rows.wfdb_record_relative))
    missing = test_ids.difference(paths)
    if len(missing):
        raise ValueError(f'SNR {snr} is missing {len(missing)} test records')

    signals = []
    for ecg_id in test_ids:
        signal, _ = wfdb.rdsamp(str(NOISY_ROOT / paths[ecg_id]))
        signals.append(signal)
    return utils.apply_standardizer(np.array(signals), scaler)


def superclass_targets_and_predictions(metadata, test_ids, mlb, y_pred):
    statements = pd.read_csv(CLEAN_ROOT / 'scp_statements.csv', index_col=0)
    superclass_names = sorted(statements.loc[statements.diagnostic == 1, 'diagnostic_class'].dropna().unique())
    labels = utils.compute_label_aggregations(metadata.copy(), str(CLEAN_ROOT) + '/', 'superdiagnostic')
    y_true = np.array([
        [superclass in labels.loc[ecg_id, 'superdiagnostic'] for superclass in superclass_names]
        for ecg_id in test_ids
    ], dtype=int)
    y_superclass_pred = np.zeros((len(y_pred), len(superclass_names)))
    for index, superclass in enumerate(superclass_names):
        class_indices = [
            position for position, scp_code in enumerate(mlb.classes_)
            if scp_code in statements.index and statements.loc[scp_code, 'diagnostic_class'] == superclass
        ]
        if not class_indices:
            raise ValueError(f'No all-task SCP predictions map to superclass {superclass}')
        y_superclass_pred[:, index] = y_pred[:, class_indices].max(axis=1)
    return superclass_names, y_true, y_superclass_pred


def main():
    print(f'Noisy mixed fastai evaluation v{SCRIPT_VERSION}')
    manifest = pd.read_csv(NOISY_ROOT / 'ptbxl_noisy_mixed_shared_manifest.csv')
    metadata = pd.read_csv(CLEAN_ROOT / 'ptbxl_database_clean_no_noise.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    test_ids = metadata.index[metadata.strat_fold == 10]

    scaler = load_pickle(EXPERIMENT_ROOT / 'data' / 'standard_scaler.pkl')
    mlb = load_pickle(EXPERIMENT_ROOT / 'data' / 'mlb.pkl')

    labels = utils.compute_label_aggregations(metadata.copy(), str(CLEAN_ROOT) + '/', 'all')
    y_test = mlb.transform(labels.loc[test_ids].all_scp.values)

    model_path = EXPERIMENT_ROOT / 'models' / 'fastai_xresnet1d101'
    model = fastai_model(
        'fastai_xresnet1d101', y_test.shape[1], 100, model_path, (1000, 12)
    )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for snr in sorted(manifest.snr_target_db.unique(), reverse=True):
        prediction_path = RESULT_ROOT / f'y_test_pred_snr_{snr}.npy'
        if prediction_path.exists():
            print(f'Loading saved SNR {snr} dB predictions...')
            y_pred = np.load(prediction_path)
        else:
            print(f'Loading SNR {snr} dB test data...')
            X_test = load_noisy_test_set(manifest, snr, test_ids, scaler)
            print(f'Predicting SNR {snr} dB...')
            y_pred = model.predict(X_test)
            np.save(prediction_path, y_pred)
        y_binary = (y_pred >= 0.5).astype(int)
        superclass_names, y_superclass, y_superclass_pred = superclass_targets_and_predictions(
            metadata, test_ids, mlb, y_pred
        )
        y_superclass_binary = (y_superclass_pred >= 0.5).astype(int)
        result = {
            'snr_db': snr,
            'n_test_records': len(test_ids),
            'macro_auc': roc_auc_score(y_test, y_pred, average='macro'),
            'label_accuracy': (y_binary == y_test).mean(),
            'exact_match_accuracy': accuracy_score(y_test, y_binary),
            'superclass_macro_auc': roc_auc_score(y_superclass, y_superclass_pred, average='macro'),
            'superclass_label_accuracy': (y_superclass_binary == y_superclass).mean(),
            'superclass_exact_match_accuracy': accuracy_score(y_superclass, y_superclass_binary),
            'superclass_macro_f1': f1_score(y_superclass, y_superclass_binary, average='macro', zero_division=0),
            'superclass_macro_recall': recall_score(y_superclass, y_superclass_binary, average='macro', zero_division=0),
        }
        for index, superclass in enumerate(superclass_names):
            result[f'superclass_recall_{superclass}'] = recall_score(
                y_superclass[:, index], y_superclass_binary[:, index], zero_division=0
            )
        results.append(result)

    results = pd.DataFrame(results).sort_values('snr_db', ascending=False)
    results.to_csv(RESULT_ROOT / 'snr_results.csv', index=False)
    print(results.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
