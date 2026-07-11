import warnings
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_EMD_FEATURES = [
    'RetainedEnergy', 'ERV', 'ERS',
    'IF_Median', 'IF_Variance', 'IF_Slope',
    'IB2_Variance', 'IB2_Slope',
    'IE12_Mean', 'IE12_Median', 'IE12_Slope',
]
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def resolve_emd_feature_columns(feature_csv_paths, feature_columns=None):
    if isinstance(feature_csv_paths, dict):
        feature_csv_paths = list(feature_csv_paths.values())
    feature_csv_paths = [Path(path) for path in feature_csv_paths]
    if not feature_csv_paths:
        raise ValueError('At least one EMD feature CSV path is required')
    candidates = CANDIDATE_EMD_FEATURES if feature_columns is None else list(feature_columns)
    available = {}
    for path in feature_csv_paths:
        if not path.exists():
            raise FileNotFoundError('EMD feature CSV does not exist: {}'.format(path))
        available[str(path)] = set(pd.read_csv(path, nrows=0).columns)
    selected = [column for column in candidates if all(column in columns for columns in available.values())]
    for column in candidates:
        missing = [path for path, columns in available.items() if column not in columns]
        if missing:
            warnings.warn('EMD feature {} is missing from: {}'.format(column, ', '.join(missing)))
    if not selected:
        raise ValueError(
            'No common EMD features remain. Candidates: {}. CSV files: {}'.format(
                candidates, [str(path) for path in feature_csv_paths]
            )
        )
    print('EMD feature columns ({}): {}'.format(len(selected), selected))
    return selected


def _metadata_ids(metadata):
    if 'ecg_id' in metadata.columns:
        return metadata.ecg_id.to_numpy()
    return metadata.index.to_numpy()


def _log_transform(features, feature_columns, enabled, log_feature_columns):
    if not enabled:
        return features
    log_feature_columns = ['RetainedEnergy'] if log_feature_columns is None else log_feature_columns
    result = features.copy()
    for column in log_feature_columns:
        if column not in feature_columns:
            continue
        index = feature_columns.index(column)
        if (result[:, :, index] < 0).any():
            raise ValueError('Cannot apply log1p to negative EMD feature {}'.format(column))
        result[:, :, index] = np.log1p(result[:, :, index])
    return result


def load_emd_features(feature_csv_path, metadata, feature_columns=None, record_ids=None,
                      missing_record_policy='drop', feature_log_transform=False,
                      log_feature_columns=None):
    if missing_record_policy not in ('drop', 'error'):
        raise ValueError('missing_record_policy must be drop or error')
    feature_csv_path = Path(feature_csv_path)
    if feature_columns is None:
        feature_columns = resolve_emd_feature_columns([feature_csv_path])
    feature_columns = list(feature_columns)
    rows = pd.read_csv(feature_csv_path, low_memory=False)
    required = ['RecordNumber', 'LeadIndex', 'Lead'] + feature_columns
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError('EMD file {} is missing columns: {}'.format(feature_csv_path, missing))
    if 'ProcessingStatus' in rows.columns:
        rows = rows[rows.ProcessingStatus == 'Success']
    valid_ids = _metadata_ids(metadata)
    if record_ids is None:
        record_ids = valid_ids
    record_ids = np.asarray(record_ids)
    rows = rows[rows.RecordNumber.isin(valid_ids)]
    rows = rows.sort_values(['RecordNumber', 'LeadIndex'])
    if rows[['RecordNumber', 'LeadIndex']].duplicated().sum():
        duplicates = rows.loc[rows[['RecordNumber', 'LeadIndex']].duplicated(keep=False), ['RecordNumber', 'LeadIndex']]
        raise ValueError('Duplicate EMD RecordNumber/LeadIndex rows: {}'.format(duplicates.head().values.tolist()))
    grouped = {record_id: group for record_id, group in rows.groupby('RecordNumber', sort=False)}
    incomplete = []
    for record_id in record_ids:
        group = grouped.get(record_id)
        if group is None or len(group) != 12 or list(group.LeadIndex) != list(range(1, 13)) or list(group.Lead) != LEAD_ORDER:
            incomplete.append(record_id)
    if incomplete and missing_record_policy == 'error':
        raise ValueError('Incomplete EMD records ({}): {}'.format(len(incomplete), incomplete[:20]))
    if incomplete:
        print('Dropping {} incomplete EMD records: {}'.format(len(incomplete), incomplete[:20]))
    incomplete_set = set(incomplete)
    kept_ids = np.array([record_id for record_id in record_ids if record_id not in incomplete_set])
    if not len(kept_ids):
        raise ValueError('No complete EMD records remain after filtering {}'.format(feature_csv_path))
    features = np.stack([
        grouped[record_id][feature_columns].to_numpy(dtype=np.float32)
        for record_id in kept_ids
    ])
    features = _log_transform(features, feature_columns, feature_log_transform, log_feature_columns)
    if not np.isfinite(features).all():
        raise ValueError('EMD features contain NaN or infinite values after loading')
    if features.shape != (len(kept_ids), 12, len(feature_columns)):
        raise AssertionError('Unexpected EMD shape {}'.format(features.shape))
    return kept_ids, features.astype(np.float32), incomplete


def fit_emd_standardizer(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_emd_standardizer(features, mean, std):
    return ((features - mean) / std).astype(np.float32)


def save_emd_standardizer(path, mean, std, feature_columns):
    np.savez(path, mean=mean, std=std, feature_columns=np.array(feature_columns), lead_order=np.array(LEAD_ORDER))
