import ast
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score

from utils import utils
from utils import data_assets


SCRIPT_VERSION = '0.1.0'
THRESHOLD = 0.5
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260711
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


def snr_bin(values):
    values = np.asarray(values)
    bins = np.full(len(values), 'outside_requested_ranges', dtype=object)
    bins[(values > 18) & (values <= 30)] = '18 < SNR <= 30'
    bins[(values > 9) & (values <= 18)] = '9 < SNR <= 18'
    bins[(values > 3) & (values <= 9)] = '3 < SNR <= 9'
    bins[(values > -3) & (values <= 3)] = '-3 < SNR <= 3'
    bins[(values >= -9) & (values <= -3)] = '-9 <= SNR <= -3'
    return bins


def safe_roc_auc(y_true, y_prob, context):
    if len(np.unique(y_true)) < 2:
        warnings.warn(f'{context}: ROC-AUC is undefined because one class is absent.')
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob, context):
    if not y_true.any():
        warnings.warn(f'{context}: PR-AUC is undefined because no positive samples are present.')
        return np.nan
    return average_precision_score(y_true, y_prob)


def macro_auc(y_true, y_prob, context):
    values = [safe_roc_auc(y_true[:, index], y_prob[:, index], f'{context}, class {index}')
              for index in range(y_true.shape[1])]
    return np.nanmean(values) if not np.all(np.isnan(values)) else np.nan


def macro_pr_auc(y_true, y_prob, context):
    values = [safe_pr_auc(y_true[:, index], y_prob[:, index], f'{context}, class {index}')
              for index in range(y_true.shape[1])]
    return np.nanmean(values) if not np.all(np.isnan(values)) else np.nan


def point_metrics(y_true, y_prob, context):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        'macro_auc': macro_auc(y_true, y_prob, context),
        'macro_pr_auc': macro_pr_auc(y_true, y_prob, context),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'samples_f1': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'label_accuracy': (y_true == y_pred).mean(),
        'exact_match': accuracy_score(y_true, y_pred),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'avg_true_labels': y_true.sum(axis=1).mean(),
        'avg_pred_labels': y_pred.sum(axis=1).mean(),
    }


def bootstrap_overall(y_true, y_prob, rng, context):
    values = {'macro_auc': [], 'macro_recall': [], 'macro_f1': [], 'exact_match': []}
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(y_true), len(y_true))
        metrics = point_metrics(y_true[indices], y_prob[indices], context)
        for name in values:
            values[name].append(metrics[name])
    return {
        f'{name}_ci_{bound}': np.nanpercentile(value, percentile)
        for name, value in values.items()
        for bound, percentile in [('low', 2.5), ('high', 97.5)]
    }


def bootstrap_class(y_true, y_pred, rng):
    recalls, f1_scores = [], []
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(y_true), len(y_true))
        recalls.append(recall_score(y_true[indices], y_pred[indices], zero_division=0))
        f1_scores.append(f1_score(y_true[indices], y_pred[indices], zero_division=0))
    return {
        'recall_ci_low': np.percentile(recalls, 2.5),
        'recall_ci_high': np.percentile(recalls, 97.5),
        'f1_ci_low': np.percentile(f1_scores, 2.5),
        'f1_ci_high': np.percentile(f1_scores, 97.5),
    }


def superclass_data(metadata, test_ids, mlb):
    statements = pd.read_csv(CLEAN_ROOT / 'scp_statements.csv', index_col=0)
    class_names = sorted(statements.loc[statements.diagnostic == 1, 'diagnostic_class'].dropna().unique())
    labels = utils.compute_label_aggregations(metadata.copy(), str(CLEAN_ROOT) + '/', 'superdiagnostic')
    y_true = np.array([
        [class_name in labels.loc[ecg_id, 'superdiagnostic'] for class_name in class_names]
        for ecg_id in test_ids
    ], dtype=int)
    prediction_indices = []
    for class_name in class_names:
        indices = [
            index for index, scp_code in enumerate(mlb.classes_)
            if scp_code in statements.index and statements.loc[scp_code, 'diagnostic_class'] == class_name
        ]
        if not indices:
            raise ValueError(f'No all-task SCP predictions map to superclass {class_name}')
        prediction_indices.append(indices)
    return class_names, y_true, prediction_indices


def main():
    print(f'Noisy superclass report generation v{SCRIPT_VERSION}')
    manifest, _ = data_assets.load_noisy_manifest(DATA_ROOT)
    clean_root = data_assets.clean_dataset_root(DATA_ROOT)
    metadata = data_assets.load_metadata(clean_root, 'ptbxl_database_clean_no_noise.csv')
    test_ids = metadata.index[metadata.strat_fold == 10]
    mlb = load_pickle(EXPERIMENT_ROOT / 'data' / 'mlb.pkl')
    class_names, y_true, prediction_indices = superclass_data(metadata, test_ids, mlb)
    rng = np.random.default_rng(RANDOM_SEED)

    samples = []
    probabilities = []
    for snr in sorted(manifest.snr_target_db.unique(), reverse=True):
        prediction_path = RESULT_ROOT / f'y_test_pred_snr_{snr}.npy'
        if not prediction_path.exists():
            raise FileNotFoundError(f'Missing predictions for target SNR {snr}: {prediction_path}')
        y_all_prob = np.load(prediction_path)
        y_prob = np.column_stack([y_all_prob[:, indices].max(axis=1) for indices in prediction_indices])
        rows = manifest[(manifest.snr_target_db == snr) & manifest.ecg_id.isin(test_ids)].set_index('ecg_id')
        rows = rows.loc[test_ids]
        if rows.snr_realized_db.isna().any():
            raise ValueError(f'Missing measured SNR for target SNR {snr}')
        y_pred = (y_prob >= THRESHOLD).astype(int)
        for index, record_id in enumerate(test_ids):
            row = {
                'record_id': record_id,
                'target_snr_db': snr,
                'measured_snr_db': rows.iloc[index].snr_realized_db,
                'snr_bin': snr_bin([rows.iloc[index].snr_realized_db])[0],
                'true_label_count': y_true[index].sum(),
                'pred_label_count': y_pred[index].sum(),
            }
            for class_index, class_name in enumerate(class_names):
                row[f'{class_name}_true'] = y_true[index, class_index]
                row[f'{class_name}_prob'] = y_prob[index, class_index]
                row[f'{class_name}_pred'] = y_pred[index, class_index]
            samples.append(row)
        probabilities.append(pd.DataFrame(y_prob, index=pd.MultiIndex.from_arrays([
            np.full(len(test_ids), snr), test_ids
        ], names=['target_snr_db', 'record_id']), columns=class_names))

    sample_predictions = pd.DataFrame(samples)
    probability_frame = pd.concat(probabilities)
    sample_predictions.to_csv(RESULT_ROOT / 'sample_predictions.csv', index=False)

    overall_rows, class_rows = [], []
    for bin_name, group in sample_predictions.groupby('snr_bin', sort=False):
        keys = pd.MultiIndex.from_arrays([group.target_snr_db, group.record_id], names=probability_frame.index.names)
        y_prob = probability_frame.loc[keys].to_numpy()
        y_group = group[[f'{class_name}_true' for class_name in class_names]].to_numpy()
        y_pred = group[[f'{class_name}_pred' for class_name in class_names]].to_numpy()
        metrics = point_metrics(y_group, y_prob, bin_name)
        metrics.update(bootstrap_overall(y_group, y_prob, rng, bin_name))
        overall_rows.append({
            'snr_bin': bin_name,
            'snr_min': group.measured_snr_db.min(),
            'snr_max': group.measured_snr_db.max(),
            'snr_mean': group.measured_snr_db.mean(),
            'snr_std': group.measured_snr_db.std(ddof=0),
            'n_samples': len(group),
            **metrics,
        })
        for index, class_name in enumerate(class_names):
            y_class, pred_class, prob_class = y_group[:, index], y_pred[:, index], y_prob[:, index]
            tp = int(((y_class == 1) & (pred_class == 1)).sum())
            fp = int(((y_class == 0) & (pred_class == 1)).sum())
            tn = int(((y_class == 0) & (pred_class == 0)).sum())
            fn = int(((y_class == 1) & (pred_class == 0)).sum())
            metrics = {
                'snr_bin': bin_name,
                'class_name': class_name,
                'positive_support': int(y_class.sum()),
                'negative_support': int((1 - y_class).sum()),
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'precision': precision_score(y_class, pred_class, zero_division=0),
                'recall': recall_score(y_class, pred_class, zero_division=0),
                'specificity': tn / (tn + fp) if tn + fp else np.nan,
                'f1': f1_score(y_class, pred_class, zero_division=0),
                'roc_auc': safe_roc_auc(y_class, prob_class, f'{bin_name}, {class_name}'),
                'pr_auc': safe_pr_auc(y_class, prob_class, f'{bin_name}, {class_name}'),
                'threshold': THRESHOLD,
                'positive_prediction_rate': pred_class.mean(),
            }
            metrics.update(bootstrap_class(y_class, pred_class, rng))
            class_rows.append(metrics)

    overall_metrics = pd.DataFrame(overall_rows)
    per_class_metrics = pd.DataFrame(class_rows)
    overall_metrics.to_csv(RESULT_ROOT / 'overall_metrics.csv', index=False)
    per_class_metrics.to_csv(RESULT_ROOT / 'per_class_metrics.csv', index=False)
    sample_counts = sample_predictions.groupby('snr_bin').size().sort_index()
    metric_counts = overall_metrics.set_index('snr_bin').n_samples.sort_index()
    if not sample_counts.equals(metric_counts):
        raise AssertionError('sample_predictions.csv SNR-bin counts do not match overall_metrics.csv')
    print('Generated sample_predictions.csv, overall_metrics.csv, and per_class_metrics.csv')


if __name__ == '__main__':
    main()
