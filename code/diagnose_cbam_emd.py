import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def bce_from_probabilities(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)).mean())


def bce_from_logits(y_true, logits):
    logits = np.asarray(logits, dtype=np.float64)
    return float(np.maximum(logits, 0).mean() - (logits * y_true).mean() + np.log1p(np.exp(-np.abs(logits))).mean())


def safe_auc(y_true, y_prob, metric):
    if len(np.unique(y_true)) < 2:
        return np.nan
    if metric == 'roc':
        return roc_auc_score(y_true, y_prob)
    return average_precision_score(y_true, y_prob)


def multilabel_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    roc = np.array([safe_auc(y_true[:, index], y_prob[:, index], 'roc') for index in range(y_true.shape[1])])
    pr = np.array([safe_auc(y_true[:, index], y_prob[:, index], 'pr') for index in range(y_true.shape[1])])
    return {
        'macro_roc_auc': float(np.nanmean(roc)),
        'micro_roc_auc': float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        'macro_pr_auc': float(np.nanmean(pr)),
        'micro_pr_auc': float(average_precision_score(y_true.ravel(), y_prob.ravel())),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'label_accuracy': float((y_true == y_pred).mean()),
        'exact_match_accuracy': float(accuracy_score(y_true, y_pred)),
        'predicted_positive_rate': float(y_pred.mean()),
        'mean_predicted_labels': float(y_pred.sum(axis=1).mean()),
        'all_zero_prediction_rate': float((y_pred.sum(axis=1) == 0).mean()),
        'per_class_roc_auc': roc,
        'per_class_pr_auc': pr,
        'per_class_f1': np.array([f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
                                  for index in range(y_true.shape[1])]),
        'per_class_prediction_rate': y_pred.mean(axis=0),
    }


def label_summary(labels, split):
    return {
        '{}_positive_rate'.format(split): float(labels.mean()),
        '{}_mean_positive_labels'.format(split): float(labels.sum(axis=1).mean()),
        '{}_median_positive_labels'.format(split): float(np.median(labels.sum(axis=1))),
        '{}_all_zero_samples'.format(split): int((labels.sum(axis=1) == 0).sum()),
        '{}_class_positive_counts'.format(split): labels.sum(axis=0).astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='exp_emd_late_fusion')
    parser.add_argument('--model', default='cbam_xresnet1d101_late_fusion')
    parser.add_argument('--output-root', default='../output')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.3])
    args = parser.parse_args()
    root = Path(args.output_root) / args.experiment
    data_root = root / 'data'
    model_root = root / 'models' / args.model
    y_train = np.load(data_root / 'y_train.npy', allow_pickle=True)
    y_val = np.load(data_root / 'y_val.npy', allow_pickle=True)
    y_test = np.load(data_root / 'y_test.npy', allow_pickle=True)
    y_prob = np.load(model_root / 'y_val_pred.npy', allow_pickle=True)
    if y_prob.shape != y_val.shape:
        raise ValueError('Validation predictions {} do not match labels {}'.format(y_prob.shape, y_val.shape))
    if not np.isfinite(y_prob).all() or y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError('y_val_pred.npy must contain finite probabilities in [0, 1]')
    if y_prob.min() >= 0.5:
        warnings.warn('All probabilities are >= 0.5. Check whether sigmoid was applied twice.')
    logits_path = model_root / 'y_val_logits.npy'
    y_logits = np.load(logits_path, allow_pickle=True) if logits_path.exists() else None
    if y_logits is not None and y_logits.shape != y_val.shape:
        raise ValueError('Validation logits {} do not match labels {}'.format(y_logits.shape, y_val.shape))
    report = {}
    report.update(label_summary(y_train, 'train'))
    report.update(label_summary(y_val, 'val'))
    report.update(label_summary(y_test, 'test'))
    all_negative = np.full_like(y_val, 1e-7, dtype=float)
    priors = np.clip(y_train.mean(axis=0), 1e-7, 1 - 1e-7)
    prior_predictions = np.tile(priors, (len(y_val), 1))
    report['model_valid_bce'] = bce_from_logits(y_val, y_logits) if y_logits is not None else bce_from_probabilities(y_val, y_prob)
    report['model_valid_bce_source'] = 'raw_logits' if y_logits is not None else 'aggregated_probabilities'
    report['all_negative_valid_bce'] = bce_from_probabilities(y_val, all_negative)
    report['class_prior_valid_bce'] = bce_from_probabilities(y_val, prior_predictions)
    report['mean_prediction_probability'] = float(y_prob.mean())
    report['probability_min'] = float(y_prob.min())
    report['probability_max'] = float(y_prob.max())
    report['probability_percentiles'] = {
        str(percentile): float(np.percentile(y_prob, percentile))
        for percentile in [1, 5, 25, 50, 75, 95, 99]
    }
    if y_logits is not None:
        report['raw_logits_min'] = float(y_logits.min())
        report['raw_logits_max'] = float(y_logits.max())
        report['raw_logits_mean'] = float(y_logits.mean())
    for threshold in args.thresholds:
        metrics = multilabel_metrics(y_val, y_prob, threshold)
        report['threshold_{}'.format(threshold)] = {
            key: value for key, value in metrics.items() if not key.startswith('per_class_')
        }
        report['threshold_{}'.format(threshold)]['valid_auc_class_count'] = int(np.isfinite(metrics['per_class_roc_auc']).sum())
        report['threshold_{}'.format(threshold)]['excluded_auc_class_count'] = int(np.isnan(metrics['per_class_roc_auc']).sum())
        per_class = pd.DataFrame({
            'class_index': range(y_val.shape[1]),
            'positive_support': y_val.sum(axis=0).astype(int),
            'negative_support': (len(y_val) - y_val.sum(axis=0)).astype(int),
            'roc_auc': metrics['per_class_roc_auc'],
            'pr_auc': metrics['per_class_pr_auc'],
            'f1': metrics['per_class_f1'],
            'prediction_rate': metrics['per_class_prediction_rate'],
        })
        per_class.to_csv(model_root / 'validation_per_class_threshold_{}.csv'.format(threshold), index=False)
    history_path = model_root / 'training_history.csv'
    if history_path.exists():
        history = pd.read_csv(history_path)
        report['best_valid_loss_epoch'] = int(history.valid_loss.idxmin())
        report['best_valid_loss'] = float(history.valid_loss.min())
        history.to_csv(model_root / 'training_history.csv', index=False)
    else:
        report['best_valid_loss_epoch'] = None
        report['best_valid_loss'] = None
    with open(model_root / 'validation_diagnosis.json', 'w') as file:
        json.dump(report, file, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
