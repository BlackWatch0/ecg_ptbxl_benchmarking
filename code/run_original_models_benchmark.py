"""Benchmark the original PTB-XL raw-waveform and Wavelet+NN models."""

import argparse
import ast
import gc
import json
import os
import pickle
import random
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import DataLoader, Dataset

from models.original_model_factory import (BENCHMARK_MODEL_NAMES, MODEL_NAMES,
                                            SE_MODEL_NAME, WAVELET_MODEL_NAME,
                                            build_original_model,
                                           build_wavelet_nn,
                                           canonical_model_name,
                                           default_learning_rate)
from utils import utils
from utils import data_assets


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SNR_LEVELS = [24, 12, 6, 0, -6]
CROP_LENGTH = 250
CROP_STRIDE = 125
WAVELET_DISPLAY_NAME = 'Wavelet+NN'
WAVELET_FEATURE_COUNT = 12 * 6 * 12
WAVELET_EPOCHS = 30
WAVELET_BATCH_SIZE = 128


class CropDataset(Dataset):
    """One random crop per training record or deterministic strided crops."""

    def __init__(self, waveforms, labels, training, crop_length=CROP_LENGTH,
                 stride=CROP_STRIDE):
        self.waveforms = waveforms
        self.labels = labels
        self.training = training
        self.crop_length = crop_length
        self.items = []
        for record_index, waveform in enumerate(waveforms):
            if len(waveform) < crop_length:
                raise ValueError('Record {} has only {} points; need {}'.format(
                    record_index, len(waveform), crop_length))
            if training:
                self.items.append((record_index, None))
            else:
                starts = list(range(0, len(waveform) - crop_length + 1, stride))
                self.items.extend((record_index, start) for start in starts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record_index, start = self.items[index]
        waveform = self.waveforms[record_index]
        if start is None:
            maximum = len(waveform) - self.crop_length
            start = 0 if maximum == 0 else random.randint(0, maximum - 1)
        crop = np.asarray(waveform[start:start + self.crop_length], dtype=np.float32).T
        label = np.asarray(self.labels[record_index], dtype=np.float32)
        return torch.from_numpy(crop), torch.from_numpy(label), record_index


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(value, file, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def cache_wavelet_features(waveforms, ids, scenario, output_root, extractor=None):
    """Load or create an ID-aligned cache of original db6/level-5 features."""
    cache_dir = output_root / 'features' / WAVELET_MODEL_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / '{}.npz'.format(scenario)
    expected_ids = np.asarray(ids)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            features = cached['features']
            cached_ids = cached['ids']
        if np.array_equal(cached_ids, expected_ids) and features.shape == (
                len(expected_ids), WAVELET_FEATURE_COUNT):
            return features
        raise ValueError('Stale or misaligned Wavelet feature cache: {}'.format(path))
    if extractor is None:
        try:
            from models.wavelet import get_ecg_features
        except ImportError as error:
            raise RuntimeError(
                'Wavelet+NN feature extraction requires PyWavelets and its scientific Python dependencies.'
            ) from error
        extractor = get_ecg_features
    features = np.asarray(extractor(waveforms), dtype=np.float32)
    if features.shape != (len(expected_ids), WAVELET_FEATURE_COUNT):
        raise ValueError('Expected Wavelet feature shape ({}, {}), got {}'.format(
            len(expected_ids), WAVELET_FEATURE_COUNT, features.shape))
    if not np.isfinite(features).all():
        raise ValueError('Wavelet feature extraction produced non-finite values for {}'.format(scenario))
    temporary = path.with_suffix('.npz.tmp')
    with open(temporary, 'wb') as file:
        np.savez_compressed(file, ids=expected_ids, features=features)
    os.replace(str(temporary), str(path))
    return features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_directories(root):
    for name in ('config', 'checkpoints', 'training_logs', 'predictions',
                 'metrics', 'errors'):
        (root / name).mkdir(parents=True, exist_ok=True)


def load_clean_data(data_root, output_root):
    clean_root = data_assets.clean_dataset_root(data_root)
    metadata_filename = next((name for name in ('ptbxl_database_clean_no_noise.csv', 'ptbxl_database.csv')
                              if (clean_root / name).exists()), None)
    if metadata_filename is None:
        raise FileNotFoundError('Missing PTB-XL metadata under {}'.format(clean_root))
    raw, metadata = utils.load_dataset(str(clean_root), 100,
                                       database_filename=metadata_filename,
                                       dataset_type='ptbxl')
    labels = utils.compute_label_aggregations(metadata, str(clean_root) + '/',
                                               'superdiagnostic')
    raw, labels, targets, mlb = utils.select_data(
        raw, labels, 'superdiagnostic', 0, str(output_root / 'config') + '/',
        class_order=CLASS_NAMES)
    if targets.shape[1] != 5 or mlb.classes_.tolist() != CLASS_NAMES:
        raise ValueError('Expected five classes in order {}, got {}'.format(
            CLASS_NAMES, mlb.classes_.tolist()))
    fold = labels.strat_fold.to_numpy()
    masks = {'train': fold <= 8, 'val': fold == 9, 'test': fold == 10}
    ids = labels.index.to_numpy()
    splits = {
        name: {'ids': ids[mask], 'ecg': raw[mask],
               'labels': targets[mask].astype(np.float32)}
        for name, mask in masks.items()
    }
    id_sets = [set(splits[name]['ids']) for name in ('train', 'val', 'test')]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise ValueError('Train/validation/test ecg_id overlap detected')
    splits['train']['ecg'], splits['val']['ecg'], splits['test']['ecg'] = \
        utils.preprocess_signals(splits['train']['ecg'], splits['val']['ecg'],
                                 splits['test']['ecg'], str(output_root / 'config') + '/')
    integrity = {
        'class_names': CLASS_NAMES,
        'train_folds': list(range(1, 9)), 'validation_fold': 9, 'test_fold': 10,
        'train_records': len(splits['train']['ids']),
        'validation_records': len(splits['val']['ids']),
        'test_records': len(splits['test']['ids']),
        'split_overlap': False,
        'test_record_ids': [int(value) for value in splits['test']['ids']],
        'standardization': 'one global scalar mean/std fit on clean folds 1-8 only',
    }
    json_dump(output_root / 'config' / 'data_integrity.json', integrity)
    return splits


def load_official_raw_data(data_root, output_root, cache_dir=None):
    official_root = data_root / 'ptbxl'
    required = ['ptbxl_database.csv', 'scp_statements.csv', 'records100']
    missing = [name for name in required if not (official_root / name).exists()]
    if missing:
        raise FileNotFoundError('Official PTB-XL data is missing {} under {}'.format(
            missing, official_root))

    metadata = pd.read_csv(official_root / 'ptbxl_database.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir).expanduser().resolve() / 'official_ptbxl_records100.npz'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            raw = cached['raw']
            cached_ids = cached['ecg_ids']
        if not np.array_equal(cached_ids, metadata.index.to_numpy()) or raw.shape != (len(metadata), 1000, 12):
            raise ValueError('Stale or misaligned official PTB-XL cache: {}'.format(cache_path))
    else:
        raw = np.asarray([wfdb.rdsamp(str(official_root / filename))[0]
                          for filename in metadata.filename_lr])
        if raw.shape != (len(metadata), 1000, 12):
            raise ValueError('Expected official records100 shape ({}, 1000, 12), got {}'.format(
                len(metadata), raw.shape))
        if cache_path:
            temporary = cache_path.with_suffix('.npz.tmp')
            with open(temporary, 'wb') as file:
                np.savez_compressed(file, raw=raw, ecg_ids=metadata.index.to_numpy())
            os.replace(str(temporary), str(cache_path))

    labels = utils.compute_label_aggregations(metadata, str(official_root) + '/',
                                               'superdiagnostic')
    raw, labels, targets, mlb = utils.select_data(
        raw, labels, 'superdiagnostic', 0, str(output_root / 'config') + '/',
        class_order=CLASS_NAMES)
    if targets.shape[1] != 5 or mlb.classes_.tolist() != CLASS_NAMES:
        raise ValueError('Expected five classes in order {}, got {}'.format(
            CLASS_NAMES, mlb.classes_.tolist()))
    fold = labels.strat_fold.to_numpy()
    masks = {'train': fold <= 8, 'val': fold == 9, 'test': fold == 10}
    ids = labels.index.to_numpy()
    splits = {
        name: {'ids': ids[mask], 'ecg': raw[mask],
               'labels': targets[mask].astype(np.float32)}
        for name, mask in masks.items()
    }
    splits['train']['ecg'], splits['val']['ecg'], splits['test']['ecg'] = \
        utils.preprocess_signals(splits['train']['ecg'], splits['val']['ecg'],
                                 splits['test']['ecg'], str(output_root / 'config') + '/')
    return splits


def _path_column(manifest):
    for column in ('wfdb_record_relative', 'record_path', 'path', 'waveform_path', 'file'):
        if column in manifest.columns:
            return column
    raise ValueError('Manifest needs a waveform path column: wfdb_record_relative, record_path, path, waveform_path, or file')


def _read_waveform(path):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        waveform = np.load(path)
    elif suffix in ('.csv', '.txt'):
        waveform = pd.read_csv(path).to_numpy()
        if waveform.shape[1] != 12:
            waveform = np.loadtxt(path, delimiter=',' if suffix == '.csv' else None)
    else:
        record = path.with_suffix('') if suffix in ('.hea', '.dat') else path
        waveform = wfdb.rdsamp(str(record))[0]
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 2:
        raise ValueError('Waveform {} must be 2-D, got {}'.format(path, waveform.shape))
    if waveform.shape[1] != 12 and waveform.shape[0] == 12:
        waveform = waveform.T
    if waveform.shape[1] != 12:
        raise ValueError('Waveform {} must have 12 leads, got {}'.format(path, waveform.shape))
    return waveform


def load_manifest_scenario(manifest_path, root, snr, expected_ids, scaler):
    manifest = pd.read_csv(manifest_path)
    if 'ecg_id' not in manifest.columns:
        raise ValueError('{} is missing column ecg_id'.format(manifest_path))
    snr_column = next((name for name in ('snr_target_db', 'snr_db', 'target_snr_db', 'snr')
                       if name in manifest.columns), None)
    if snr_column is None:
        raise ValueError('{} needs an SNR column'.format(manifest_path))
    path_column = _path_column(manifest)
    numeric_snr = pd.to_numeric(manifest[snr_column], errors='coerce')
    rows = manifest[numeric_snr == snr].copy()
    rows.ecg_id = pd.to_numeric(rows.ecg_id, errors='raise').astype(np.int64)
    duplicates = sorted(int(x) for x in rows.loc[rows.ecg_id.duplicated(False), 'ecg_id'].unique())
    rows = rows.set_index('ecg_id')
    missing = [int(value) for value in expected_ids if int(value) not in rows.index]
    if duplicates or missing:
        raise ValueError('SNR {} alignment failed; duplicate ecg_id={}, missing ecg_id={}'.format(
            snr, duplicates[:20], missing[:20]))
    rows = rows.loc[[int(value) for value in expected_ids]]
    waveforms = np.array([_read_waveform(root / str(value)) for value in rows[path_column]])
    standardized = utils.apply_standardizer(waveforms, scaler).astype(np.float32)
    return standardized, {
        'number_of_test_records': len(expected_ids),
        'number_of_matched_ecg_records': len(rows),
        'duplicate_record_ids': duplicates, 'missing_record_ids': missing,
        'record_order_matches_test_ecg_ids': True, 'label_consistency': True,
        'manifest': str(manifest_path), 'root': str(root), 'target_snr_db': snr,
    }


def external_scenarios(args, test, scaler):
    supplied = [args.noisy_manifest, args.noisy_root,
                args.denoised_manifest, args.denoised_root]
    if any(supplied) and not all(supplied):
        raise ValueError('Supply all of --noisy-manifest, --noisy-root, --denoised-manifest, and --denoised-root')
    scenarios = []
    if not all(supplied):
        return scenarios
    for source, manifest, root in (
            ('noisy', Path(args.noisy_manifest), Path(args.noisy_root)),
            ('denoised', Path(args.denoised_manifest), Path(args.denoised_root))):
        for snr in SNR_LEVELS:
            ecg, integrity = load_manifest_scenario(manifest, root, snr,
                                                    test['ids'], scaler)
            label = 'snrm{}'.format(abs(snr)) if snr < 0 else 'snr{}'.format(snr)
            scenarios.append(('{}_{}'.format(source, label), snr, ecg, integrity))
    return scenarios


def apply_data_config(args):
    if not args.data_config:
        return
    with open(Path(args.data_config).expanduser()) as file:
        data_config = json.load(file)
    manifests = data_config.get('manifests', {})
    roots = data_config.get('wfdb_roots', {})
    required = [('noisy_manifest', manifests.get('noisy')),
                ('denoised_manifest', manifests.get('denoised')),
                ('noisy_root', roots.get('noisy')),
                ('denoised_root', roots.get('denoised'))]
    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError('Data config is missing {}'.format(missing))
    for name, value in required:
        current = getattr(args, name)
        if current and Path(current).expanduser().resolve() != Path(value).expanduser().resolve():
            raise ValueError('--data-config conflicts with --{}'.format(name.replace('_', '-')))
        setattr(args, name, value)


def smoke_test_data_config(args, output_root, device):
    """Cheap contract check used before launching a full accelerator run."""
    if args.data_config:
        with open(Path(args.data_config).expanduser()) as file:
            value = json.load(file)
        if value.get('snrs_db') != SNR_LEVELS:
            raise ValueError('Data config SNRs must be {}'.format(SNR_LEVELS))
        expected_count = int(value.get('fold10_record_count', 0))
        for condition in ('noisy', 'denoised'):
            frame = pd.read_csv(value['manifests'][condition])
            snr_column = next((name for name in ('snr_target_db', 'snr_db', 'target_snr_db', 'snr')
                               if name in frame), None)
            if snr_column is None or any((pd.to_numeric(frame[snr_column]) == snr).sum() != expected_count
                                         for snr in SNR_LEVELS):
                raise ValueError('{} manifest does not have complete five-SNR coverage'.format(condition))
    outputs = {}
    for name in MODEL_NAMES:
        model = build_original_model(name).to(device).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 12, CROP_LENGTH, device=device))
        if output.shape != (1, 5) or not torch.isfinite(output).all():
            raise ValueError('{} smoke forward failed'.format(name))
        outputs[name] = list(output.shape)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    wavelet = build_wavelet_nn(WAVELET_FEATURE_COUNT)
    wavelet_output = np.asarray(wavelet(np.zeros((1, WAVELET_FEATURE_COUNT), dtype=np.float32)))
    if wavelet_output.shape != (1, 5) or not np.isfinite(wavelet_output).all():
        raise ValueError('Wavelet+NN smoke forward failed')
    outputs[WAVELET_DISPLAY_NAME] = list(wavelet_output.shape)
    json_dump(output_root / 'config' / 'smoke_test.json', {
        'status': 'passed', 'model_outputs': outputs,
        'data_config': args.data_config, 'device': str(device),
    })
    print('Smoke test passed for six raw-waveform models and Wavelet+NN.')


def build_loader(split, training, batch_size, shuffle=None, crop_length=CROP_LENGTH,
                 num_workers=0):
    dataset = CropDataset(split['ecg'], split['labels'], training=training,
                          crop_length=crop_length)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=training if shuffle is None else shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def run_one_batch_smoke_test(args, data_root, device):
    official_root = data_root / 'ptbxl'
    required = ['ptbxl_database.csv', 'scp_statements.csv', 'records100']
    missing = [name for name in required if not (official_root / name).exists()]
    if missing:
        raise FileNotFoundError('Official PTB-XL data is missing {} under {}'.format(
            missing, official_root))
    if len(args.models) != 1:
        raise ValueError('--one-batch-smoke-test requires exactly one model')

    model_name = canonical_model_name(args.models[0])
    if model_name != SE_MODEL_NAME:
        raise ValueError('--one-batch-smoke-test requires {}'.format(SE_MODEL_NAME))

    metadata = pd.read_csv(official_root / 'ptbxl_database.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    labels = utils.compute_label_aggregations(metadata, str(official_root) + '/',
                                               'superdiagnostic')
    labels = labels[labels.superdiagnostic_len > 0]
    train_labels = labels[labels.strat_fold <= 8].head(args.batch_size)
    if len(train_labels) != args.batch_size:
        raise ValueError('Need {} labeled training records, found {}'.format(
            args.batch_size, len(train_labels)))

    mlb = MultiLabelBinarizer(classes=CLASS_NAMES)
    mlb.fit([CLASS_NAMES])
    targets = mlb.transform(train_labels.superdiagnostic).astype(np.float32)
    waveforms = np.asarray([
        wfdb.rdsamp(str(official_root / filename))[0]
        for filename in train_labels.filename_lr
    ])
    if waveforms.shape[1:] != (1000, 12):
        raise ValueError('Expected records100 waveforms shaped [N, 1000, 12], got {}'.format(
            waveforms.shape))
    if not np.isfinite(waveforms).all():
        raise FloatingPointError('Official PTB-XL batch contains NaN or Inf')

    with tempfile.TemporaryDirectory() as temporary:
        standardized, _, _ = utils.preprocess_signals(waveforms, [], [], temporary + os.sep)
    split = {'ecg': standardized, 'labels': targets}
    loader = build_loader(split, True, args.batch_size, shuffle=False, crop_length=1000)
    ecg, target, _ = next(iter(loader))
    ecg, target = ecg.to(device), target.to(device)

    model = build_original_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=default_learning_rate(model_name),
                                 weight_decay=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    logits = model(ecg)
    loss = criterion(logits, target)
    if not torch.isfinite(loss):
        raise FloatingPointError('One-batch smoke test produced a non-finite loss')
    parameter_before = next(model.parameters()).detach().clone()
    loss.backward()
    backward_success = any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    optimizer_step_success = not torch.equal(parameter_before, next(model.parameters()).detach())

    result = {
        'data_root': str(official_root), 'model_name': model_name,
        'loss_type': 'BCEWithLogitsLoss', 'input_shape': list(ecg.shape),
        'label_shape': list(target.shape), 'output_shape': list(logits.shape),
        'label_names': CLASS_NAMES, 'multi_hot_labels': target.detach().cpu().tolist(),
        'loss': float(loss.detach().cpu()), 'backward_success': backward_success,
        'optimizer_step_success': optimizer_step_success,
        'contains_nan': bool(torch.isnan(ecg).any()),
        'contains_inf': bool(torch.isinf(ecg).any()), 'device': str(device),
    }
    if device.type == 'cuda':
        result['gpu_name'] = torch.cuda.get_device_name(device)
        result['gpu_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2)
    print(json.dumps(result, indent=2))


def _autocast(device, enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(str(temporary), str(path))


def _atomic_csv_save(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _rng_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_model(model, train_loader, valid_loader, config, device, best_path,
                last_path, history_path, resume=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['learning_rate'], epochs=config['epochs'],
        steps_per_epoch=len(train_loader))
    criterion = torch.nn.BCEWithLogitsLoss()
    amp = bool(config.get('mixed_precision', True) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if amp else None
    start_epoch, best_epoch, best_loss, elapsed, history = 0, -1, np.inf, 0.0, []
    if resume and last_path.exists():
        state = load_torch_checkpoint(last_path, device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = int(state['epoch'])
        best_epoch, best_loss = int(state['best_epoch']), float(state['best_valid_loss'])
        elapsed = float(state.get('training_time_seconds', 0.0))
        same_schedule = (
            state.get('steps_per_epoch') == len(train_loader) and
            state.get('scheduler', {}).get('total_steps') == config['epochs'] * len(train_loader)
        )
        if same_schedule:
            scheduler.load_state_dict(state['scheduler'])
        else:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(start_epoch * len(train_loader)):
                optimizer.step()
                scheduler.step()
        if scaler is not None and state.get('scaler') is not None:
            scaler.load_state_dict(state['scaler'])
        _restore_rng(state.get('rng_state'))
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict('records')[:start_epoch]
        print('Resuming at true epoch {}/{}'.format(start_epoch, config['epochs']))
    started = time.time()
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        train_losses = []
        for batch_index, (ecg, labels, _) in enumerate(train_loader, start=1):
            ecg, labels = ecg.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model(ecg)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss at epoch {}'.format(epoch + 1))
            if loss.detach().item() > 100:
                raise FloatingPointError('Abnormally large training loss at epoch {}'.format(epoch + 1))
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config['model_name'] in ('lstm', 'lstm_bidir'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config['model_name'] in ('lstm', 'lstm_bidir'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
                optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))
            if batch_index % 100 == 0:
                message = '{} epoch {}/{} batch {}/{} mean_loss={:.6f}'.format(
                    config['model_name'], epoch + 1, config['epochs'], batch_index,
                    len(train_loader), np.mean(train_losses))
                if device.type == 'cuda':
                    message += ' gpu_peak_mib={:.2f}'.format(
                        torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                print(message)
        model.eval()
        valid_losses = []
        print('{} validation start'.format(config['model_name']))
        with torch.no_grad():
            for ecg, labels, _ in valid_loader:
                logits = model(ecg.to(device))
                loss = criterion(logits, labels.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite validation loss at epoch {}'.format(epoch + 1))
                valid_losses.append(float(loss.cpu()))
        print('{} validation complete'.format(config['model_name']))
        valid_loss = float(np.mean(valid_losses))
        train_loss = float(np.mean(train_losses))
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        _atomic_csv_save(pd.DataFrame(history), history_path)
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch + 1
            _atomic_torch_save({'model': model.state_dict(), 'epoch': best_epoch,
                                'best_valid_loss': best_loss, 'config': config}, best_path)
        total_elapsed = elapsed + time.time() - started
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch + 1, 'best_epoch': best_epoch,
            'best_valid_loss': best_loss, 'training_time_seconds': total_elapsed,
            'steps_per_epoch': len(train_loader), 'rng_state': _rng_state(),
            'config': config,
        }
        _atomic_torch_save(state, last_path)
        print('{} epoch {}/{} train_loss={:.6f} valid_loss={:.6f}'.format(
            config['model_name'], epoch + 1, config['epochs'], train_loss, valid_loss))
    return best_epoch, best_loss, elapsed + time.time() - started


def predict_crops(model, loader, device, record_count):
    model.eval()
    record_probabilities = [[] for _ in range(record_count)]
    labels_by_record = [None] * record_count
    started = time.time()
    with torch.no_grad():
        for ecg, labels, record_indices in loader:
            probabilities = torch.sigmoid(model(ecg.to(device))).cpu().numpy()
            for probability, label, record_index in zip(probabilities, labels.numpy(),
                                                         record_indices.numpy()):
                record_probabilities[int(record_index)].append(probability)
                labels_by_record[int(record_index)] = label
    if any(not values for values in record_probabilities):
        raise ValueError('At least one record had no evaluation crop')
    probabilities = np.vstack([np.max(values, axis=0) for values in record_probabilities])
    elapsed_ms = (time.time() - started) * 1000 / record_count
    return probabilities, np.vstack(labels_by_record), elapsed_ms


def threshold_results(y_true, probabilities):
    values = np.round(np.arange(.10, .91, .01), 2)
    curve = [(float(value), float(f1_score(y_true, probabilities >= value,
                                            average='macro', zero_division=0)))
             for value in values]
    global_threshold = max(curve, key=lambda item: item[1])[0]
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        scores = [(float(value), float(f1_score(y_true[:, index],
                                                probabilities[:, index] >= value,
                                                zero_division=0)))
                  for value in values]
        per_class[name] = max(scores, key=lambda item: item[1])[0]
    return global_threshold, per_class, curve


def _safe_metric(function, y_true, y_prob):
    return float(function(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan


def metric_rows(y_true, probabilities, prediction, metadata):
    roc = [_safe_metric(roc_auc_score, y_true[:, i], probabilities[:, i]) for i in range(5)]
    pr = [_safe_metric(average_precision_score, y_true[:, i], probabilities[:, i]) for i in range(5)]
    challenge = utils.challenge_metrics(y_true, prediction, beta1=2, beta2=2)
    overall = dict(metadata)
    overall.update({
        'macro_roc_auc': float(np.nanmean(roc)),
        'macro_auc': float(np.nanmean(roc)),
        'micro_roc_auc': float(roc_auc_score(y_true.ravel(), probabilities.ravel())),
        'macro_pr_auc': float(np.nanmean(pr)),
        'micro_pr_auc': float(average_precision_score(y_true.ravel(), probabilities.ravel())),
        'macro_f1': float(f1_score(y_true, prediction, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, prediction, average='micro', zero_division=0)),
        'samples_f1': float(f1_score(y_true, prediction, average='samples', zero_division=0)),
        'label_accuracy': float((y_true == prediction).mean()),
        'exact_match_accuracy': float((y_true == prediction).all(axis=1).mean()),
        'predicted_positive_rate': float(prediction.mean()),
        'mean_predicted_labels': float(prediction.sum(axis=1).mean()),
        'all_zero_prediction_rate': float((prediction.sum(axis=1) == 0).mean()),
        'F_beta_macro': float(challenge['F_beta_macro']),
        'G_beta_macro': float(challenge['G_beta_macro']),
    })
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        true, pred = y_true[:, index], prediction[:, index]
        tn = int(((true == 0) & (pred == 0)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        per_class.append(dict(
            metadata, class_name=name, roc_auc=roc[index], pr_auc=pr[index],
            precision=float(precision_score(true, pred, zero_division=0)),
            recall=float(recall_score(true, pred, zero_division=0)),
            specificity=float(tn / (tn + fp)) if tn + fp else np.nan,
            f1=float(f1_score(true, pred, zero_division=0)),
            support_positive=int(true.sum()), support_negative=int((true == 0).sum()),
            predicted_positive_count=int(pred.sum())))
    return overall, per_class


def save_predictions(path, ids, y_true, probabilities, prediction):
    frame = pd.DataFrame({'record_id': ids, 'ecg_id': ids})
    for index, name in enumerate(CLASS_NAMES):
        frame['true_' + name] = y_true[:, index].astype(int)
        frame['prob_' + name] = probabilities[:, index]
        frame['pred_' + name] = prediction[:, index].astype(int)
    frame.to_csv(path, index=False)


def count_parameters(model):
    return (sum(value.numel() for value in model.parameters()),
            sum(value.numel() for value in model.parameters() if value.requires_grad))


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            'Wavelet+NN requires TensorFlow. Use the Colab wrapper or install a compatible tensorflow package.'
        ) from error
    return tf


def train_wavelet_classifier(train_features, train_labels, val_features, val_labels,
                             checkpoint_dir, history_path, seed, resume=False,
                             evaluate_only=False):
    """Fit or restore the original Wavelet+NN classifier and train-only scaler."""
    tf = _tensorflow()
    scaler_path = checkpoint_dir / 'standard_scaler.pkl'
    best_path = checkpoint_dir / 'best_loss_model.keras'
    last_path = checkpoint_dir / 'last_model.keras'
    resume_paths = (scaler_path, best_path, last_path, history_path)
    continuing = resume and all(path.exists() for path in resume_paths)
    if resume and any(path.exists() for path in resume_paths) and not continuing:
        missing = [str(path) for path in resume_paths if not path.exists()]
        raise RuntimeError('Incomplete Wavelet resume state; missing {}'.format(missing))
    if (continuing or evaluate_only) and scaler_path.exists():
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
    elif evaluate_only:
        raise FileNotFoundError('Wavelet train-only scaler is missing: {}'.format(scaler_path))
    else:
        scaler = StandardScaler().fit(train_features)
        temporary = scaler_path.with_suffix('.pkl.tmp')
        with open(temporary, 'wb') as file:
            pickle.dump(scaler, file)
        os.replace(str(temporary), str(scaler_path))
    train_scaled = scaler.transform(train_features).astype(np.float32)
    val_scaled = scaler.transform(val_features).astype(np.float32)
    initial_epoch = 0
    if (continuing or evaluate_only) and history_path.exists():
        history = pd.read_csv(history_path)
        if len(history):
            initial_epoch = int(history.epoch.max())
    if evaluate_only:
        if not best_path.exists():
            raise FileNotFoundError('Wavelet best checkpoint is missing: {}'.format(best_path))
    elif continuing:
        model = tf.keras.models.load_model(last_path)
    else:
        tf.keras.utils.set_random_seed(seed)
        model = build_wavelet_nn(train_scaled.shape[1])
        initial_epoch = 0
    started = time.time()
    if not evaluate_only and initial_epoch < WAVELET_EPOCHS:
        class HistoryCheckpoint(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                row = pd.DataFrame([{
                    'model_name': WAVELET_DISPLAY_NAME, 'seed': seed,
                    'epoch': epoch + 1, 'train_loss': float(logs['loss']),
                    'valid_loss': float(logs['val_loss']),
                }])
                existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
                updated = pd.concat([existing, row], ignore_index=True).drop_duplicates(
                    'epoch', keep='last').sort_values('epoch')
                temporary = history_path.with_suffix('.csv.tmp')
                updated.to_csv(temporary, index=False)
                os.replace(str(temporary), str(history_path))

        previous_best = None
        if continuing:
            previous_best = float(pd.read_csv(history_path).valid_loss.min())
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(best_path), monitor='val_loss', mode='min', save_best_only=True,
                initial_value_threshold=previous_best),
            tf.keras.callbacks.ModelCheckpoint(str(last_path), save_best_only=False),
            HistoryCheckpoint(),
        ]
        model.fit(train_scaled, train_labels,
                  validation_data=(val_scaled, val_labels),
                  epochs=WAVELET_EPOCHS, initial_epoch=initial_epoch,
                  batch_size=WAVELET_BATCH_SIZE, callbacks=callbacks, verbose=2)
    training_seconds = time.time() - started
    if not best_path.exists():
        raise FileNotFoundError('Wavelet best-validation-loss checkpoint is missing: {}'.format(best_path))
    best_model = tf.keras.models.load_model(best_path)
    if not history_path.exists() or not len(pd.read_csv(history_path)):
        best_epoch = -1
        best_loss = float(best_model.evaluate(val_scaled, val_labels, verbose=0))
    else:
        history = pd.read_csv(history_path)
        best_row = history.loc[history.valid_loss.idxmin()]
        best_epoch, best_loss = int(best_row.epoch), float(best_row.valid_loss)
    return best_model, scaler, best_epoch, best_loss, training_seconds, best_path


def predict_wavelet(model, scaler, features):
    started = time.time()
    probabilities = np.asarray(model.predict(
        scaler.transform(features).astype(np.float32),
        batch_size=WAVELET_BATCH_SIZE, verbose=0))
    return probabilities, (time.time() - started) * 1000 / len(features)


def run_wavelet(seed, splits, scenarios, output_root, args):
    run_name = WAVELET_MODEL_NAME
    checkpoint_dir = output_root / 'checkpoints' / run_name / 'seed_{}'.format(seed)
    metric_dir = output_root / 'metrics' / run_name
    prediction_dir = output_root / 'predictions' / run_name / 'seed_{}'.format(seed)
    log_dir = output_root / 'training_logs' / run_name
    for directory in (checkpoint_dir, metric_dir, prediction_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / 'seed_{}.csv'.format(seed)
    set_seed(seed)
    train_features = cache_wavelet_features(
        splits['train']['ecg'], splits['train']['ids'], 'train', output_root)
    val_features = cache_wavelet_features(
        splits['val']['ecg'], splits['val']['ids'], 'validation', output_root)
    model, scaler, best_epoch, best_loss, training_seconds, best_path = \
        train_wavelet_classifier(
            train_features, splits['train']['labels'], val_features, splits['val']['labels'],
            checkpoint_dir, history_path, seed, resume=args.resume,
            evaluate_only=args.evaluate_only)
    val_prob, _ = predict_wavelet(model, scaler, val_features)
    val_y = splits['val']['labels']
    global_threshold, per_class_thresholds, curve = threshold_results(val_y, val_prob)
    json_dump(checkpoint_dir / 'thresholds.json', {
        'threshold_0.5': .5, 'best_global_threshold': global_threshold,
        'per_class_thresholds': per_class_thresholds,
        'selected_on': 'clean validation fold 9',
    })
    pd.DataFrame(curve, columns=['threshold', 'validation_macro_f1']).to_csv(
        metric_dir / 'seed_{}_threshold_search.csv'.format(seed), index=False)
    per_class_array = np.array([per_class_thresholds[name] for name in CLASS_NAMES])
    save_predictions(prediction_dir / 'validation_predictions.csv', splits['val']['ids'],
                     val_y, val_prob, (val_prob >= per_class_array).astype(int))
    all_scenarios = [('clean', None, splits['test']['ecg'], {
        'number_of_test_records': len(splits['test']['ids']),
        'number_of_matched_ecg_records': len(splits['test']['ids']),
        'duplicate_record_ids': [], 'missing_record_ids': [],
        'record_order_matches_test_ecg_ids': True, 'label_consistency': True,
    })] + scenarios
    parameter_count = int(model.count_params())
    trainable_count = int(sum(np.prod(value.shape) for value in model.trainable_weights))
    all_overall, all_per_class = [], []
    for scenario, snr, ecg, integrity in all_scenarios:
        features = cache_wavelet_features(
            ecg, splits['test']['ids'], scenario, output_root)
        probabilities, inference_ms = predict_wavelet(model, scaler, features)
        y_test = splits['test']['labels']
        json_dump(metric_dir / 'seed_{}_{}_integrity.json'.format(seed, scenario), integrity)
        for strategy, threshold in (
                ('threshold_0.5', .5),
                ('best_global_threshold', global_threshold),
                ('per_class_thresholds', per_class_array)):
            prediction = (probabilities >= threshold).astype(int)
            metadata = {
                'experiment_name': WAVELET_DISPLAY_NAME,
                'model_name': WAVELET_DISPLAY_NAME, 'seed': seed,
                'ecg_scenario': scenario, 'feature_scenario': scenario,
                'target_snr_db': snr, 'threshold_strategy': strategy,
                'threshold': json.dumps(threshold.tolist()) if hasattr(threshold, 'tolist') else threshold,
                'best_epoch': best_epoch, 'best_valid_loss': best_loss,
                'training_time_seconds': training_seconds,
                'parameter_count': parameter_count,
                'trainable_parameter_count': trainable_count,
                'inference_time_per_sample_ms': inference_ms,
                'actual_batch_size': WAVELET_BATCH_SIZE,
                'checkpoint_path': str(best_path), 'crop_length': np.nan,
                'crop_stride': np.nan, 'crop_aggregation': 'not_used',
                'status': 'completed',
            }
            overall, per_class = metric_rows(y_test, probabilities, prediction, metadata)
            all_overall.append(overall)
            all_per_class.extend(per_class)
            strategy_slug = strategy.replace('threshold_', '')
            save_predictions(prediction_dir / 'test_predictions_{}_{}.csv'.format(
                scenario, strategy_slug), splits['test']['ids'], y_test,
                probabilities, prediction)
            if strategy == 'per_class_thresholds':
                save_predictions(prediction_dir / 'test_predictions_{}.csv'.format(scenario),
                                 splits['test']['ids'], y_test, probabilities, prediction)
    pd.DataFrame(all_overall).to_csv(metric_dir / 'seed_{}.csv'.format(seed), index=False)
    pd.DataFrame(all_per_class).to_csv(metric_dir / 'seed_{}_per_class.csv'.format(seed), index=False)
    model_info = {
        'experiment_name': WAVELET_DISPLAY_NAME, 'model_name': WAVELET_DISPLAY_NAME,
        'seed': seed, 'best_epoch': best_epoch, 'best_valid_loss': best_loss,
        'training_time_seconds': training_seconds, 'parameter_count': parameter_count,
        'trainable_parameter_count': trainable_count,
        'actual_batch_size': WAVELET_BATCH_SIZE, 'checkpoint_path': str(best_path),
        'status': 'completed',
    }
    json_dump(checkpoint_dir / 'model_info.json', model_info)
    pd.DataFrame([model_info]).to_csv(
        metric_dir / 'seed_{}_complexity.csv'.format(seed), index=False)


def run_one(model_name, seed, splits, scenarios, config, output_root, device, args):
    model_name = canonical_model_name(model_name)
    run_name = model_name
    checkpoint_dir = output_root / 'checkpoints' / run_name / 'seed_{}'.format(seed)
    metric_dir = output_root / 'metrics' / run_name
    prediction_dir = output_root / 'predictions' / run_name / 'seed_{}'.format(seed)
    log_dir = output_root / 'training_logs' / run_name
    for directory in (checkpoint_dir, metric_dir, prediction_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / 'checkpoint.pth'
    last_path = checkpoint_dir / 'last_checkpoint.pth'
    history_path = log_dir / 'seed_{}.csv'.format(seed)
    set_seed(seed)
    actual_batch_size = config['batch_size']
    training_seconds = None
    if not args.evaluate_only:
        trained = False
        if config['crop_length'] == 1000:
            candidates = [config['batch_size'], 16, 8]
        else:
            candidates = [config['batch_size'], 128, 64, 32, 16]
        if args.resume and last_path.exists():
            previous = load_torch_checkpoint(last_path, torch.device('cpu'))
            previous_batch = previous.get('config', {}).get('actual_batch_size')
            if previous_batch:
                candidates.insert(0, int(previous_batch))
        for candidate in dict.fromkeys(candidates):
            if candidate > config['batch_size']:
                continue
            model = build_original_model(model_name).to(device)
            try:
                train_loader = build_loader(
                    splits['train'], True, candidate, crop_length=config['crop_length'],
                    num_workers=args.num_workers)
                valid_loader = build_loader(
                    splits['val'], False, candidate, crop_length=config['crop_length'],
                    num_workers=args.num_workers)
                run_config = dict(config, model_name=model_name,
                                  learning_rate=args.learning_rate or default_learning_rate(model_name),
                                  actual_batch_size=candidate, seed=seed)
                best_epoch, best_loss, training_seconds = train_model(
                    model, train_loader, valid_loader, run_config, device,
                    best_path, last_path, history_path,
                    resume=args.resume or candidate != config['batch_size'])
                actual_batch_size, trained = candidate, True
                break
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower() or candidate == min(candidates):
                    raise
                print('CUDA OOM for {} at batch {}; retrying smaller.'.format(model_name, candidate))
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        if not trained:
            raise RuntimeError('Training did not complete for {}'.format(model_name))
    if not best_path.exists():
        raise FileNotFoundError('Best checkpoint is missing: {}'.format(best_path))
    model = build_original_model(model_name).to(device)
    best_state = load_torch_checkpoint(best_path, device)
    model.load_state_dict(best_state['model'])
    best_epoch = int(best_state['epoch'])
    best_loss = float(best_state['best_valid_loss'])
    if last_path.exists():
        last_state = load_torch_checkpoint(last_path, torch.device('cpu'))
        training_seconds = last_state.get('training_time_seconds', training_seconds)
        actual_batch_size = last_state.get('config', {}).get('actual_batch_size', actual_batch_size)
    parameter_count, trainable_count = count_parameters(model)
    val_loader = build_loader(
        splits['val'], False, actual_batch_size, crop_length=config['crop_length'],
        num_workers=args.num_workers)
    val_prob, val_y, _ = predict_crops(model, val_loader, device, len(splits['val']['ids']))
    val_prediction = (val_prob >= .5).astype(int)
    val_roc = [_safe_metric(roc_auc_score, val_y[:, index], val_prob[:, index])
               for index in range(len(CLASS_NAMES))]
    val_f1 = [float(f1_score(val_y[:, index], val_prediction[:, index], zero_division=0))
              for index in range(len(CLASS_NAMES))]
    validation_metrics = {
        'threshold': .5,
        'loss': best_loss,
        'macro_roc_auc': float(np.nanmean(val_roc)),
        'macro_f1': float(f1_score(val_y, val_prediction, average='macro', zero_division=0)),
        'per_class': {name: {'roc_auc': val_roc[index], 'f1': val_f1[index]}
                      for index, name in enumerate(CLASS_NAMES)},
    }
    json_dump(metric_dir / 'seed_{}_validation_metrics.json'.format(seed), validation_metrics)
    save_predictions(prediction_dir / 'validation_predictions_threshold_0_5.csv',
                     splits['val']['ids'], val_y, val_prob, val_prediction)
    global_threshold, per_class_thresholds, curve = threshold_results(val_y, val_prob)
    json_dump(checkpoint_dir / 'thresholds.json', {
        'threshold_0.5': .5, 'best_global_threshold': global_threshold,
        'per_class_thresholds': per_class_thresholds,
        'selected_on': 'clean validation fold 9',
    })
    pd.DataFrame(curve, columns=['threshold', 'validation_macro_f1']).to_csv(
        metric_dir / 'seed_{}_threshold_search.csv'.format(seed), index=False)
    if args.skip_test_evaluation:
        return
    per_class_array = np.array([per_class_thresholds[name] for name in CLASS_NAMES])
    save_predictions(prediction_dir / 'validation_predictions.csv', splits['val']['ids'],
                     val_y, val_prob, (val_prob >= per_class_array).astype(int))
    all_scenarios = [('clean', None, splits['test']['ecg'], {
        'number_of_test_records': len(splits['test']['ids']),
        'number_of_matched_ecg_records': len(splits['test']['ids']),
        'duplicate_record_ids': [], 'missing_record_ids': [],
        'record_order_matches_test_ecg_ids': True, 'label_consistency': True,
    })] + scenarios
    all_overall, all_per_class = [], []
    for scenario, snr, ecg, integrity in all_scenarios:
        split = {'ids': splits['test']['ids'], 'labels': splits['test']['labels'], 'ecg': ecg}
        loader = build_loader(
            split, False, actual_batch_size, crop_length=config['crop_length'],
            num_workers=args.num_workers)
        probabilities, y_test, inference_ms = predict_crops(model, loader, device, len(split['ids']))
        json_dump(metric_dir / 'seed_{}_{}_integrity.json'.format(seed, scenario), integrity)
        strategies = [('threshold_0.5', .5),
                      ('best_global_threshold', global_threshold),
                      ('per_class_thresholds', per_class_array)]
        for strategy, threshold in strategies:
            prediction = (probabilities >= threshold).astype(int)
            metadata = {
                'experiment_name': model_name, 'model_name': model_name, 'seed': seed,
                'ecg_scenario': scenario, 'feature_scenario': 'not_used',
                'target_snr_db': snr, 'threshold_strategy': strategy,
                'threshold': json.dumps(threshold.tolist()) if hasattr(threshold, 'tolist') else threshold,
                'best_epoch': best_epoch, 'best_valid_loss': best_loss,
                'training_time_seconds': training_seconds,
                'parameter_count': parameter_count,
                'trainable_parameter_count': trainable_count,
                'inference_time_per_sample_ms': inference_ms,
                'actual_batch_size': actual_batch_size, 'checkpoint_path': str(best_path),
                'crop_length': CROP_LENGTH, 'crop_stride': CROP_STRIDE,
                'crop_aggregation': 'max_probability', 'status': 'completed',
            }
            overall, per_class = metric_rows(y_test, probabilities, prediction, metadata)
            all_overall.append(overall)
            all_per_class.extend(per_class)
            strategy_slug = strategy.replace('threshold_', '')
            save_predictions(prediction_dir / 'test_predictions_{}_{}.csv'.format(
                scenario, strategy_slug), split['ids'], y_test, probabilities, prediction)
            if strategy == 'per_class_thresholds':
                save_predictions(prediction_dir / 'test_predictions_{}.csv'.format(scenario),
                                 split['ids'], y_test, probabilities, prediction)
    pd.DataFrame(all_overall).to_csv(metric_dir / 'seed_{}.csv'.format(seed), index=False)
    pd.DataFrame(all_per_class).to_csv(metric_dir / 'seed_{}_per_class.csv'.format(seed), index=False)
    model_info = {
        'experiment_name': model_name, 'model_name': model_name, 'seed': seed,
        'best_epoch': best_epoch, 'best_valid_loss': best_loss,
        'training_time_seconds': training_seconds, 'parameter_count': parameter_count,
        'trainable_parameter_count': trainable_count,
        'actual_batch_size': actual_batch_size, 'checkpoint_path': str(best_path),
        'status': 'completed',
    }
    json_dump(checkpoint_dir / 'model_info.json', model_info)
    pd.DataFrame([model_info]).to_csv(metric_dir / 'seed_{}_complexity.csv'.format(seed), index=False)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(REPO_ROOT / 'data'))
    parser.add_argument('--data-config', help='Normalized manifest configuration from prepare_original_models_benchmark_data.py')
    parser.add_argument('--output-dir', default=str(REPO_ROOT / 'results' / 'original_models_benchmark'))
    parser.add_argument('--models', nargs='+', default=list(BENCHMARK_MODEL_NAMES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--crop-length', type=int, default=CROP_LENGTH)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--official-raw-data', action='store_true')
    parser.add_argument('--cache-dir')
    parser.add_argument('--skip-test-evaluation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--one-batch-smoke-test', action='store_true')
    parser.add_argument('--no-mixed-precision', action='store_true')
    parser.add_argument('--noisy-manifest')
    parser.add_argument('--noisy-root')
    parser.add_argument('--denoised-manifest')
    parser.add_argument('--denoised-root')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    apply_data_config(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if args.smoke_test and args.one_batch_smoke_test:
        raise ValueError('--smoke-test and --one-batch-smoke-test cannot be combined')
    if args.crop_length < 1:
        raise ValueError('--crop-length must be positive')
    unknown = [name for name in args.models
               if canonical_model_name(name) not in BENCHMARK_MODEL_NAMES + (SE_MODEL_NAME,)]
    if unknown:
        raise ValueError('Unknown models: {}'.format(unknown))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.one_batch_smoke_test:
        run_one_batch_smoke_test(args, data_root, device)
        return
    prepare_directories(output_root)
    config = {
        'task': 'superdiagnostic', 'class_names': CLASS_NAMES,
        'train_folds': list(range(1, 9)), 'validation_fold': 9, 'test_fold': 10,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'weight_decay': 1e-2, 'optimizer': 'Adam', 'scheduler': 'OneCycleLR',
        'loss': 'BCEWithLogitsLoss', 'mixed_precision': not args.no_mixed_precision,
        'crop_length': args.crop_length, 'validation_test_stride': CROP_STRIDE,
        'crop_aggregation': 'max_probability', 'models': args.models, 'seeds': args.seeds,
        'wavelet_nn': {
            'wavelet': 'db6', 'decomposition_level': 5,
            'dense_units': 128, 'dropout': .25, 'activation': 'relu',
            'output_activation': 'sigmoid', 'optimizer': 'Adamax',
            'loss': 'binary_crossentropy', 'epochs': WAVELET_EPOCHS,
            'batch_size': WAVELET_BATCH_SIZE,
            'checkpoint_selection': 'minimum clean validation loss',
            'feature_scaler': 'StandardScaler fit on clean training folds 1-8 only',
        },
    }
    json_dump(output_root / 'config' / 'resolved_config.json', config)
    obsolete_status = output_root / 'config' / 'wavelet_nn_status.json'
    if obsolete_status.exists():
        obsolete_status.unlink()
    print('Device: {}, output: {}'.format(device, output_root))
    if args.smoke_test:
        smoke_test_data_config(args, output_root, device)
        return
    if args.official_raw_data:
        splits = load_official_raw_data(data_root, output_root, args.cache_dir)
    else:
        splits = load_clean_data(data_root, output_root)
    with open(output_root / 'config' / 'standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    scenarios = [] if args.skip_test_evaluation else external_scenarios(args, splits['test'], scaler)
    completed = {}
    failures = []
    for seed in args.seeds:
        for requested_name in args.models:
            model_name = canonical_model_name(requested_name)
            key = '{}_{}'.format(model_name, seed)
            try:
                if model_name == WAVELET_MODEL_NAME:
                    run_wavelet(seed, splits, scenarios, output_root, args)
                else:
                    run_one(model_name, seed, splits, scenarios, config, output_root, device, args)
                completed[key] = 'completed'
            except Exception:
                error_path = output_root / 'errors' / '{}_seed_{}.log'.format(model_name, seed)
                error_path.write_text(traceback.format_exc())
                completed[key] = 'failed'
                failures.append((model_name, seed, error_path))
                print('FAILED {} seed {}; see {}'.format(model_name, seed, error_path))
            json_dump(output_root / 'completed_models.json', completed)
    if failures:
        raise RuntimeError('Benchmark failures: {}'.format(
            ', '.join('{} seed {} ({})'.format(*failure) for failure in failures)))


if __name__ == '__main__':
    main()
