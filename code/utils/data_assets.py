"""Shared, path-safe loaders for prepared PTB-XL waveform and feature assets."""
import ast
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


def as_path(path):
    return Path(path).expanduser().resolve()


def clean_dataset_root(data_root):
    root = as_path(data_root)
    return root if (root / 'ptbxl_database_clean_no_noise.csv').exists() or (root / 'ptbxl_database.csv').exists() else root / 'ptbxl_clean_no_noise'


def noisy_dataset_root(data_root):
    root = as_path(data_root)
    return root if (root / 'ptbxl_noisy_mixed_shared_manifest.csv').exists() else root / 'ptbxl_noisy_mixed_shared'


def load_metadata(root, filename=None, dataset_type='ptbxl'):
    root = as_path(root)
    if dataset_type == 'ptbxl':
        candidates = [filename] if filename else ['ptbxl_database_clean_no_noise.csv', 'ptbxl_database.csv']
    elif dataset_type == 'ICBEB':
        candidates = [filename or 'icbeb_database.csv']
    else:
        raise ValueError("Unknown dataset_type {!r}. Expected 'ptbxl' or 'ICBEB'.".format(dataset_type))
    path = next((root / name for name in candidates if name and (root / name).exists()), None)
    if path is None:
        raise FileNotFoundError('Metadata not found under {} (tried {})'.format(root, candidates))
    metadata = pd.read_csv(path, index_col='ecg_id')
    if 'scp_codes' in metadata:
        metadata.scp_codes = metadata.scp_codes.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
    return metadata


def load_noisy_manifest(data_root):
    root = noisy_dataset_root(data_root)
    path = root / 'ptbxl_noisy_mixed_shared_manifest.csv'
    if not path.exists():
        raise FileNotFoundError('Noisy PTB-XL manifest not found: {}'.format(path))
    return pd.read_csv(path), root


def load_manifest_waveforms(manifest, root, snr, record_ids):
    required = {'ecg_id', 'snr_target_db', 'wfdb_record_relative'}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError('Noisy manifest is missing {}'.format(sorted(missing)))
    rows = manifest[manifest.snr_target_db == snr].set_index('ecg_id')
    duplicate_ids = rows.index[rows.index.duplicated()].unique().tolist()
    missing_ids = [int(value) for value in record_ids if value not in rows.index]
    if duplicate_ids or missing_ids:
        raise ValueError('SNR {} alignment failed; duplicate IDs={}, missing IDs={}'.format(
            snr, duplicate_ids[:20], missing_ids[:20]))
    rows = rows.loc[list(record_ids)]
    return np.array([wfdb.rdsamp(str(Path(root) / path))[0] for path in rows.wfdb_record_relative])


def load_waveforms(metadata, root, sampling_rate, dataset_type='ptbxl'):
    if sampling_rate not in (100, 500):
        raise ValueError('sampling_rate must be 100 or 500, got {}'.format(sampling_rate))
    root = as_path(root)
    cache = root / 'raw{}.npy'.format(sampling_rate)
    if cache.exists():
        return np.load(str(cache), allow_pickle=True)
    if dataset_type == 'ptbxl':
        column = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        records = [root / value for value in metadata[column]]
    elif dataset_type == 'ICBEB':
        records = [root / 'records{}'.format(sampling_rate) / str(value) for value in metadata.index]
    else:
        raise ValueError('Unsupported dataset_type {}'.format(dataset_type))
    data = np.array([wfdb.rdsamp(str(record))[0] for record in tqdm(records)])
    # Existing caches were pickle payloads with an .npy extension; retain that format.
    with open(cache, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)
    return data


def load_dataset(root, sampling_rate, filename=None, dataset_type='ptbxl'):
    metadata = load_metadata(root, filename, dataset_type)
    return load_waveforms(metadata, root, sampling_rate, dataset_type), metadata


def resolve_emd_paths(data_root):
    root = as_path(data_root) / 'emd_features'
    paths = {'clean': root / 'original' / 'PTBXL_Batch_Original_EMD_reduced_features.csv'}
    legacy_names = {'snr24': 'mixed_snr24_MAT_Batch_EMD_reduced_features.csv',
                    'snr12': 'mixed_snr12_MAT_Batch_EMD_reduced_features.csv',
                    'snr6': 'mixed_snr6_DenoisedCSV_EMD_reduced_features.csv',
                    'snr0': 'mixed_snr0_DenoisedCSV_EMD_reduced_features.csv',
                    'snrm6': 'mixed_snrm6_MAT_Batch_EMD_reduced_features.csv'}
    for scenario, legacy_name in legacy_names.items():
        active = root / 'ptbxl_original_database_plus_mixed' / ('mixed_' + scenario) / ('mixed_' + scenario + '_plus_mixed_EMD_Features_reduced_features.csv')
        legacy = root / ('mixed_' + scenario) / legacy_name
        paths[scenario] = active if active.exists() else legacy
    return paths
