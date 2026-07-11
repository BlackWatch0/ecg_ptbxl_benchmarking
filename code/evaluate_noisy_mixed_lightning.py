import ast
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from models.lightning_checkpoint_models import load_checkpoint_model
from models.timeseries_utils import TimeseriesDatasetCrops, ToTensor, aggregate_predictions
from utils import utils


SCRIPT_VERSION = '0.1.0'
DATA_ROOT = Path('../data')
OUTPUT_ROOT = Path('../output')
NOISY_ROOT = DATA_ROOT / 'ptbxl_noisy_mixed_shared'
CLEAN_ROOT = DATA_ROOT / 'ptbxl_clean_no_noise'
EXPERIMENT_ROOT = OUTPUT_ROOT / 'exp0'
RESULT_ROOT = OUTPUT_ROOT / 'noisy_mixed_shared' / 'lightning'
MODELS = {
    'lenet_all': ('lenet', 'all', 71),
    'lenet_superdiagnostic': ('lenet', 'superdiagnostic', 5),
    'lstm_all': ('lstm', 'all', 71),
    'lstm_superdiagnostic': ('lstm', 'superdiagnostic', 5),
    'resnet_all': ('resnet', 'all', 71),
    'resnet_superdiagnostic': ('resnet', 'superdiagnostic', 5),
    'inception_all': ('inception', 'all', 71),
    'inception_superdiagnostic': ('inception', 'superdiagnostic', 5),
    'xresnet_all': ('xresnet', 'all', 71),
    'xresnet_superdiagnostic': ('xresnet', 'superdiagnostic', 5),
}
SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


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
    signals = [wfdb.rdsamp(str(NOISY_ROOT / paths[ecg_id]))[0] for ecg_id in test_ids]
    return utils.apply_standardizer(np.array(signals), scaler)


def predict(model, X, num_classes, batch_size=128, device='cpu'):
    frame = pd.DataFrame({
        'data': range(len(X)),
        'label': [np.zeros(num_classes) for _ in range(len(X))],
    })
    frame['data_length'] = [len(signal) for signal in X]
    dataset = TimeseriesDatasetCrops(
        frame, 250, num_classes=num_classes, chunk_length=250,
        min_chunk_length=250, stride=125, transforms=[ToTensor()],
        annotation=False, col_lbl='label', npy_data=[signal.astype(np.float32) for signal in X],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for data, _ in loader:
            predictions.append(model(data.to(device)).cpu().numpy())
    predictions = 1.0 / (1.0 + np.exp(-np.concatenate(predictions, axis=0)))
    return aggregate_predictions(predictions, idmap=dataset.get_id_mapping(), aggregate_fn=np.amax)


def labels_for_task(metadata, task):
    labels = utils.compute_label_aggregations(metadata.copy(), str(CLEAN_ROOT) + '/', task)
    _, labels, y, _ = utils.select_data(np.empty(len(labels), dtype=object), labels, task, 0, '/tmp/')
    test_mask = labels.strat_fold == 10
    return labels.index[test_mask], y[test_mask]


def main():
    if tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2]) < (2, 4):
        raise RuntimeError('This script requires PyTorch 2.4 or newer to read the Lightning checkpoints.')

    print(f'Noisy mixed Lightning evaluation v{SCRIPT_VERSION}')
    manifest = pd.read_csv(NOISY_ROOT / 'ptbxl_noisy_mixed_shared_manifest.csv')
    metadata = pd.read_csv(CLEAN_ROOT / 'ptbxl_database_clean_no_noise.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    scaler = load_pickle(EXPERIMENT_ROOT / 'data' / 'standard_scaler.pkl')
    targets = {task: labels_for_task(metadata, task) for task in {'all', 'superdiagnostic'}}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for snr in sorted(manifest.snr_target_db.unique(), reverse=True):
        X_test = {}
        for task, (test_ids, _) in targets.items():
            print(f'Loading {task}, SNR {snr} dB test data...')
            X_test[task] = load_noisy_test_set(manifest, snr, test_ids, scaler)
        for model_name, (architecture, task, num_classes) in MODELS.items():
            model_root = RESULT_ROOT / model_name
            model_root.mkdir(parents=True, exist_ok=True)
            prediction_path = model_root / f'y_test_pred_snr_{snr}.npy'
            if prediction_path.exists():
                print(f'Loading saved {model_name}, SNR {snr} dB predictions...')
                y_pred = np.load(prediction_path)
            else:
                checkpoint = OUTPUT_ROOT / model_name / 'checkpoints' / 'best_model.ckpt'
                print(f'Predicting {model_name}, SNR {snr} dB...')
                model = load_checkpoint_model(checkpoint, architecture, num_classes)
                y_pred = predict(model, X_test[task], num_classes)
                np.save(prediction_path, y_pred)
            test_ids, y_test = targets[task]
            y_binary = (y_pred >= 0.5).astype(int)
            result = {
                'model': model_name,
                'task': task,
                'snr_db': snr,
                'n_test_records': len(test_ids),
                'macro_auc': roc_auc_score(y_test, y_pred, average='macro'),
                'label_accuracy': (y_binary == y_test).mean(),
                'exact_match_accuracy': accuracy_score(y_test, y_binary),
            }
            if task == 'superdiagnostic':
                result['macro_f1'] = f1_score(y_test, y_binary, average='macro', zero_division=0)
                result['macro_recall'] = recall_score(y_test, y_binary, average='macro', zero_division=0)
                for index, superclass in enumerate(SUPERCLASSES):
                    result[f'recall_{superclass}'] = recall_score(
                        y_test[:, index], y_binary[:, index], zero_division=0
                    )
            results.append(result)

    results = pd.DataFrame(results).sort_values(['model', 'snr_db'], ascending=[True, False])
    results.to_csv(RESULT_ROOT / 'snr_results.csv', index=False)
    print(results.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
