import argparse
import ast
import gc
import json
import os
import random
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

from models.cbam_xresnet1d import build_model
from utils import utils
from utils.emd_features import (apply_emd_standardizer, fit_emd_standardizer,
                                load_emd_features, resolve_emd_feature_columns)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SNR_SCENARIOS = [('clean', None), ('snr24', 24), ('snr12', 12), ('snr6', 6),
                 ('snr0', 0), ('snrm6', -6)]
EXPERIMENTS = {
    'xresnet1d101_baseline': dict(use_cbam=False, use_se=False, use_emd=False, fusion_type='none'),
    'cbam_xresnet1d101': dict(use_cbam=True, use_se=False, use_emd=False, fusion_type='none'),
    'se_xresnet1d101': dict(use_cbam=False, use_se=True, use_emd=False, fusion_type='none'),
    'xresnet1d101_emd_late_fusion': dict(use_cbam=False, use_se=False, use_emd=True, fusion_type='concat'),
    'cbam_xresnet1d101_emd_late_fusion': dict(use_cbam=True, use_se=False, use_emd=True, fusion_type='concat'),
    'se_xresnet1d101_emd_late_fusion': dict(use_cbam=False, use_se=True, use_emd=True, fusion_type='concat'),
}
DEFAULT_CONFIG = {
    'task': 'superdiagnostic', 'num_classes': 5, 'class_names': CLASS_NAMES,
    'epochs': 50, 'batch_size': 128, 'learning_rate': 1e-2, 'weight_decay': 1e-2,
    'optimizer': 'Adam', 'loss': 'BCEWithLogitsLoss',
    'monitor': 'valid_loss', 'save_best_only': True, 'threshold_search': True,
    'seeds': [42], 'snr_levels': ['clean', 24, 12, 6, 0, -6],
    'mixed_precision': True, 'data_root': 'data',
    'output_dir': '/content/drive/MyDrive/ECG/ablation_results', 'experiments': list(EXPERIMENTS),
}


class ECGDataset(Dataset):
    def __init__(self, ecg, labels, emd=None):
        self.ecg = ecg
        self.labels = labels
        self.emd = emd

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ecg = torch.from_numpy(np.asarray(self.ecg[index], dtype=np.float32).T)
        label = torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32))
        if self.emd is None:
            return ecg, label
        return ecg, torch.from_numpy(np.asarray(self.emd[index], dtype=np.float32)), label


def parse_args():
    parser = argparse.ArgumentParser(description='Run reproducible PTB-XL CBAM/EMD ablations.')
    parser.add_argument('--config', default=str(REPO_ROOT / 'configs' / 'ablation_cbam_emd.yaml'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def read_config(args):
    config = dict(DEFAULT_CONFIG)
    path = Path(args.config)
    if path.exists():
        with open(path) as file:
            config.update(yaml.safe_load(file) or {})
    else:
        raise FileNotFoundError('Configuration file does not exist: {}'.format(path))
    if args.experiments:
        config['experiments'] = args.experiments
    if args.seeds:
        config['seeds'] = args.seeds
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if config['task'] != 'superdiagnostic' or config['class_names'] != CLASS_NAMES:
        raise ValueError('This runner is fixed to 5-class superdiagnostic {}'.format(CLASS_NAMES))
    unknown = set(config['experiments']).difference(EXPERIMENTS)
    if unknown:
        raise ValueError('Unknown experiments: {}'.format(sorted(unknown)))
    return config


def prepare_directories(output_root):
    names = ['config', 'checkpoints', 'training_logs', 'predictions', 'metrics',
             'figures', 'errors', 'final_report']
    for name in names:
        (output_root / name).mkdir(parents=True, exist_ok=True)


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(value, file, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_output_dir(config):
    requested = Path(config['output_dir']).expanduser()
    drive = Path('/content/drive/MyDrive/ECG/ablation_results')
    if str(requested).startswith('/content/drive') and not Path('/content/drive/MyDrive').exists():
        fallback = REPO_ROOT / 'results' / 'ablation_study'
        print('Google Drive is unavailable; using {}'.format(fallback.resolve()))
        return fallback.resolve()
    if drive.parent.exists() and str(requested) == '../results/ablation_study':
        return drive
    return (requested if requested.is_absolute() else REPO_ROOT / requested).resolve()


def eprint_environment(device):
    print('PyTorch: {}, CUDA available: {}'.format(torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print('GPU: {}, total memory: {:.1f} GB'.format(props.name, props.total_memory / 1024 ** 3))


def load_data(config, output_root):
    configured_root = Path(config['data_root'])
    data_root = (configured_root if configured_root.is_absolute() else REPO_ROOT / configured_root).resolve()
    clean_root = data_root / 'ptbxl_clean_no_noise'
    metadata_path = clean_root / 'ptbxl_database_clean_no_noise.csv'
    if not metadata_path.exists():
        raise FileNotFoundError('Missing clean PTB-XL metadata: {}'.format(metadata_path))
    raw, metadata = utils.load_dataset(str(clean_root), 100,
                                       database_filename=metadata_path.name, dataset_type='ptbxl')
    labels = utils.compute_label_aggregations(metadata, str(clean_root) + '/', 'superdiagnostic')
    raw, labels, targets, mlb = utils.select_data(raw, labels, 'superdiagnostic', 0,
                                                    str(output_root / 'config') + '/',
                                                    class_order=CLASS_NAMES)
    if mlb.classes_.tolist() != CLASS_NAMES or targets.shape[1] != 5:
        raise ValueError('Expected 5-class labels {}, got {}'.format(CLASS_NAMES, mlb.classes_.tolist()))
    ids = labels.index.to_numpy()
    split = {}
    for name, folds in [('train', labels.strat_fold <= 8), ('val', labels.strat_fold == 9),
                        ('test', labels.strat_fold == 10)]:
        split[name] = dict(ids=ids[folds.to_numpy()], ecg=raw[folds.to_numpy()],
                           labels=targets[folds.to_numpy()].astype(np.float32))
    sets = [set(split[name]['ids']) for name in ('train', 'val', 'test')]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Train/validation/test record IDs overlap')
    train, val, test = split['train'], split['val'], split['test']
    train['ecg'], val['ecg'], test['ecg'] = utils.preprocess_signals(
        train['ecg'], val['ecg'], test['ecg'], str(output_root / 'config') + '/')
    emd_paths = {
        'clean': data_root / 'emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv',
        'snr24': data_root / 'emd_features/mixed_snr24/mixed_snr24_MAT_Batch_EMD_reduced_features.csv',
        'snr12': data_root / 'emd_features/mixed_snr12/mixed_snr12_MAT_Batch_EMD_reduced_features.csv',
        'snr6': data_root / 'emd_features/mixed_snr6/mixed_snr6_DenoisedCSV_EMD_reduced_features.csv',
        'snr0': data_root / 'emd_features/mixed_snr0/mixed_snr0_DenoisedCSV_EMD_reduced_features.csv',
        'snrm6': data_root / 'emd_features/mixed_snrm6/mixed_snrm6_MAT_Batch_EMD_reduced_features.csv',
    }
    existing = [path for path in emd_paths.values() if path.exists()]
    if not (emd_paths['clean'].exists()):
        raise FileNotFoundError('Clean EMD file is required: {}'.format(emd_paths['clean']))
    columns = resolve_emd_feature_columns(existing)
    for name in ('train', 'val', 'test'):
        found_ids, features, incomplete = load_emd_features(emd_paths['clean'], labels, columns,
                                                              split[name]['ids'], 'error')
        if incomplete or not np.array_equal(found_ids, split[name]['ids']):
            raise ValueError('Clean EMD is not exactly aligned for {}'.format(name))
        split[name]['emd_raw'] = features
    mean, std = fit_emd_standardizer(train['emd_raw'])
    for name in split:
        split[name]['emd'] = apply_emd_standardizer(split[name]['emd_raw'], mean, std)
    np.savez(output_root / 'config' / 'emd_scaler.npz', mean=mean, std=std,
             feature_columns=np.array(columns))
    json_dump(output_root / 'config' / 'data_integrity.json', {
        'class_names': CLASS_NAMES, 'train_records': len(train['ids']), 'val_records': len(val['ids']),
        'test_records': len(test['ids']), 'split_overlap': False, 'emd_feature_columns': columns,
        'test_record_ids': [int(x) for x in test['ids']],
    })
    print('Data: train/val/test = {}/{}/{}, EMD features = {}'.format(
        len(train['ids']), len(val['ids']), len(test['ids']), len(columns)))
    return dict(root=data_root, labels=labels, splits=split, emd_paths=emd_paths,
                emd_columns=columns, emd_mean=mean, emd_std=std)


def load_smoke_data(config, output_root):
    """Load real records without depending on the optional raw100.npy cache."""
    configured_root = Path(config['data_root'])
    data_root = (configured_root if configured_root.is_absolute() else REPO_ROOT / configured_root).resolve()
    clean_root = data_root / 'ptbxl_clean_no_noise'
    metadata = pd.read_csv(clean_root / 'ptbxl_database_clean_no_noise.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    labels = utils.compute_label_aggregations(metadata, str(clean_root) + '/', 'superdiagnostic')
    placeholder = np.empty(len(labels), dtype=object)
    _, labels, targets, mlb = utils.select_data(placeholder, labels, 'superdiagnostic', 0,
                                                 str(output_root / 'config') + '/', CLASS_NAMES)
    if mlb.classes_.tolist() != CLASS_NAMES:
        raise ValueError('Smoke test found incorrect class order {}'.format(mlb.classes_.tolist()))
    emd_paths = {
        'clean': data_root / 'emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv',
        'snr24': data_root / 'emd_features/mixed_snr24/mixed_snr24_MAT_Batch_EMD_reduced_features.csv',
        'snr12': data_root / 'emd_features/mixed_snr12/mixed_snr12_MAT_Batch_EMD_reduced_features.csv',
        'snr6': data_root / 'emd_features/mixed_snr6/mixed_snr6_DenoisedCSV_EMD_reduced_features.csv',
        'snr0': data_root / 'emd_features/mixed_snr0/mixed_snr0_DenoisedCSV_EMD_reduced_features.csv',
        'snrm6': data_root / 'emd_features/mixed_snrm6/mixed_snrm6_MAT_Batch_EMD_reduced_features.csv',
    }
    columns = resolve_emd_feature_columns([path for path in emd_paths.values() if path.exists()])
    split = {}
    masks = {'train': labels.strat_fold <= 8, 'val': labels.strat_fold == 9, 'test': labels.strat_fold == 10}
    all_ids = []
    for name, mask in masks.items():
        ids = labels.index[mask].to_numpy()
        all_ids.append(set(ids))
        record_id = ids[0]
        waveform = wfdb.rdsamp(str(clean_root / metadata.loc[record_id, 'filename_lr']))[0][np.newaxis, ...]
        found_ids, features, incomplete = load_emd_features(emd_paths['clean'], labels, columns,
                                                              np.array([record_id]), 'error')
        if incomplete or not np.array_equal(found_ids, [record_id]):
            raise ValueError('Smoke EMD alignment failed for {} record {}'.format(name, record_id))
        split[name] = dict(ids=np.array([record_id]), ecg=waveform.astype(np.float32),
                           emd=features, labels=targets[mask.to_numpy()][:1].astype(np.float32))
    if all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2] or all_ids[1] & all_ids[2]:
        raise ValueError('Smoke test found fold leakage')
    mean, std = fit_emd_standardizer(split['train']['emd'])
    for value in split.values():
        value['emd'] = apply_emd_standardizer(value['emd'], mean, std)
    print('Smoke real-data split IDs: train={}, val={}, test={}'.format(
        split['train']['ids'][0], split['val']['ids'][0], split['test']['ids'][0]))
    return dict(root=data_root, labels=labels, splits=split, emd_paths=emd_paths,
                emd_columns=columns, emd_mean=mean, emd_std=std)


def load_noisy_ecg(bundle, scenario, snr):
    test = bundle['splits']['test']
    noisy_root = bundle['root'] / 'ptbxl_noisy_mixed_shared'
    manifest_path = noisy_root / 'ptbxl_noisy_mixed_shared_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError('Missing noisy manifest: {}'.format(manifest_path))
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest.snr_target_db == snr].set_index('ecg_id')
    duplicates = rows.index[rows.index.duplicated()].unique().tolist()
    missing = [int(x) for x in test['ids'] if x not in rows.index]
    if duplicates or missing:
        raise ValueError('{} waveform alignment failed; duplicate IDs: {}, missing IDs: {}'.format(
            scenario, duplicates[:20], missing[:20]))
    rows = rows.loc[test['ids']]
    records = np.array([wfdb.rdsamp(str(noisy_root / path))[0] for path in rows.wfdb_record_relative])
    scaler_path = Path(bundle['scaler_path'])
    import pickle
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return utils.apply_standardizer(records, scaler), dict(
        number_of_test_records=len(test['ids']), number_of_matched_ecg_records=len(rows),
        duplicate_record_ids=duplicates, missing_record_ids=missing, label_consistency=True)


def scenario_emd(bundle, scenario):
    test = bundle['splits']['test']
    requested = bundle['emd_paths'][scenario]
    source = 'matched_{}'.format(scenario) if scenario != 'clean' else 'clean_original'
    feature_scenario = scenario
    if not requested.exists():
        print('WARNING: matched EMD file missing for {}; using clean/original EMD upper bound.'.format(scenario))
        requested = bundle['emd_paths']['clean']
        source, feature_scenario = 'clean_original', 'clean'
    found_ids, features, incomplete = load_emd_features(requested, bundle['labels'], bundle['emd_columns'],
                                                          test['ids'], 'error')
    if incomplete or not np.array_equal(found_ids, test['ids']):
        raise ValueError('EMD alignment failed for {} from {}'.format(scenario, requested))
    return apply_emd_standardizer(features, bundle['emd_mean'], bundle['emd_std']), source, feature_scenario, {
        'number_of_matched_emd_records': len(found_ids), 'emd_file': str(requested),
        'emd_duplicate_record_ids': [], 'emd_missing_record_ids': []
    }


def build_loader(split, use_emd, batch_size, shuffle=False):
    return DataLoader(ECGDataset(split['ecg'], split['labels'], split['emd'] if use_emd else None),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def forward(model, batch, device, use_emd):
    if use_emd:
        ecg, emd, labels = batch
        return model(ecg.to(device), emd.to(device)), labels.to(device)
    ecg, labels = batch
    return model(ecg.to(device)), labels.to(device)


def predict(model, loader, device, use_emd):
    model.eval()
    probs, targets = [], []
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            logits, labels = forward(model, batch, device, use_emd)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())
    probabilities = np.vstack(probs)
    return probabilities, np.vstack(targets), (time.time() - started) * 1000 / len(probabilities)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, valid_loader, config, device, checkpoint, history_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],
                                                    epochs=config['epochs'], steps_per_epoch=len(train_loader))
    criterion = torch.nn.BCEWithLogitsLoss()
    amp = bool(config['mixed_precision'] and device.type == 'cuda' and hasattr(torch.cuda, 'amp'))
    if amp and hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler() if amp else None
    best_loss, best_epoch, history = np.inf, -1, []
    started = time.time()
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if amp:
                autocast = torch.amp.autocast('cuda') if hasattr(torch, 'amp') else torch.cuda.amp.autocast()
                with autocast:
                    logits, labels = forward(model, batch, device, config['use_emd'])
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, labels = forward(model, batch, device, config['use_emd'])
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                logits, labels = forward(model, batch, device, config['use_emd'])
                valid_losses.append(float(criterion(logits, labels).cpu()))
        train_loss, valid_loss = float(np.mean(train_losses)), float(np.mean(valid_losses))
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'valid_loss': valid_loss,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print('{} epoch {}/{} train_loss={:.6f} valid_loss={:.6f}'.format(
            config['experiment_name'], epoch + 1, config['epochs'], train_loss, valid_loss))
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch + 1
            torch.save({'model': model.state_dict(), 'epoch': best_epoch,
                        'best_valid_loss': best_loss, 'config': config}, checkpoint)
    return best_epoch, best_loss, time.time() - started


def safe_metric(function, y_true, y_prob):
    return float(function(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan


def threshold_results(y_true, probabilities):
    values = np.round(np.arange(.10, .91, .01), 2)
    curve = []
    for threshold in values:
        curve.append((float(threshold), float(f1_score(y_true, probabilities >= threshold,
                                                       average='macro', zero_division=0))))
    best_global = max(curve, key=lambda item: item[1])[0]
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        scores = [(float(t), float(f1_score(y_true[:, index], probabilities[:, index] >= t,
                                            zero_division=0))) for t in values]
        per_class[name] = max(scores, key=lambda item: item[1])[0]
    return best_global, per_class, curve


def metric_rows(y_true, probabilities, prediction, metadata):
    roc = [safe_metric(roc_auc_score, y_true[:, i], probabilities[:, i]) for i in range(5)]
    pr = [safe_metric(average_precision_score, y_true[:, i], probabilities[:, i]) for i in range(5)]
    overall = dict(metadata)
    overall.update({
        'macro_roc_auc': float(np.nanmean(roc)), 'micro_roc_auc': float(roc_auc_score(y_true.ravel(), probabilities.ravel())),
        'macro_pr_auc': float(np.nanmean(pr)), 'micro_pr_auc': float(average_precision_score(y_true.ravel(), probabilities.ravel())),
        'macro_f1': float(f1_score(y_true, prediction, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, prediction, average='micro', zero_division=0)),
        'samples_f1': float(f1_score(y_true, prediction, average='samples', zero_division=0)),
        'label_accuracy': float((y_true == prediction).mean()),
        'exact_match_accuracy': float((y_true == prediction).all(axis=1).mean()),
        'predicted_positive_rate': float(prediction.mean()),
        'mean_predicted_labels': float(prediction.sum(axis=1).mean()),
        'all_zero_prediction_rate': float((prediction.sum(axis=1) == 0).mean()),
    })
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        true, pred = y_true[:, index], prediction[:, index]
        tn = int(((true == 0) & (pred == 0)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        per_class.append(dict(metadata, class_name=name, roc_auc=roc[index], pr_auc=pr[index],
                              precision=float(precision_score(true, pred, zero_division=0)),
                              recall=float(recall_score(true, pred, zero_division=0)),
                              specificity=float(tn / (tn + fp)) if tn + fp else np.nan,
                              f1=float(f1_score(true, pred, zero_division=0)),
                              support_positive=int(true.sum()), support_negative=int((true == 0).sum()),
                              predicted_positive_count=int(pred.sum())))
    return overall, per_class


def save_predictions(path, ids, y_true, probabilities, prediction):
    result = pd.DataFrame({'record_id': ids})
    for index, name in enumerate(CLASS_NAMES):
        result['true_' + name] = y_true[:, index].astype(int)
        result['prob_' + name] = probabilities[:, index]
        result['pred_' + name] = prediction[:, index].astype(int)
    result.to_csv(path, index=False)


def experiment_complete(root, name, seed):
    checkpoint = root / 'checkpoints' / name / 'seed_{}'.format(seed) / 'checkpoint.pth'
    history = root / 'training_logs' / name / 'seed_{}.csv'.format(seed)
    metrics = root / 'metrics' / name / 'seed_{}.csv'.format(seed)
    predictions = root / 'predictions' / name / 'seed_{}'.format(seed)
    return checkpoint.exists() and history.exists() and metrics.exists() and all(
        (predictions / 'test_predictions_{}.csv'.format(scenario)).exists()
        for scenario, _ in SNR_SCENARIOS)


def smoke_test(bundle, config, device):
    split = bundle['splits']['train']
    if split['ecg'][0].shape != (1000, 12) or split['emd'].shape[1:] != (12, len(bundle['emd_columns'])):
        raise ValueError('Unexpected input shapes ECG {} EMD {}'.format(split['ecg'][0].shape, split['emd'].shape))
    parameter_counts = {}
    for name, spec in EXPERIMENTS.items():
        model = build_model('xresnet1d101', 5, **spec, emd_features=len(bundle['emd_columns'])).to(device)
        model.train()
        ecg = torch.from_numpy(split['ecg'][:1].transpose(0, 2, 1).astype(np.float32)).to(device)
        emd = torch.from_numpy(split['emd'][:1].astype(np.float32)).to(device)
        logits = model(ecg, emd if spec['use_emd'] else None)
        if logits.shape != (1, 5) or not torch.isfinite(logits).all():
            raise ValueError('{} smoke forward failed with {}'.format(name, tuple(logits.shape)))
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.from_numpy(split['labels'][:1]).to(device))
        if not torch.isfinite(loss):
            raise ValueError('{} smoke loss is not finite'.format(name))
        loss.backward()
        parameter_counts[name] = count_parameters(model)[0]
        if spec['use_se']:
            se_modules = [module for module in model.modules() if module.__class__.__name__ == 'SqueezeExcitation1d']
            if not se_modules:
                raise ValueError('{} does not contain SE blocks'.format(name))
            sample_channels = se_modules[0].excitation[0].in_channels
            scale = se_modules[0].scale(torch.randn(2, sample_channels, 17, device=device))
            if scale.shape != (2, sample_channels, 1) or not torch.isfinite(scale).all():
                raise ValueError('{} SE scale check failed: {}'.format(name, tuple(scale.shape)))
            if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for module in se_modules for parameter in module.parameters()):
                raise ValueError('{} SE backward check failed'.format(name))
        print('Smoke passed: {} -> {}'.format(name, tuple(logits.shape)))
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if parameter_counts['se_xresnet1d101'] <= parameter_counts['xresnet1d101_baseline']:
        raise ValueError('SE parameter count must exceed baseline')
    if parameter_counts['se_xresnet1d101_emd_late_fusion'] <= parameter_counts['xresnet1d101_emd_late_fusion']:
        raise ValueError('SE+EMD parameter count must exceed baseline+EMD')


def load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    return state


def run_one(name, seed, bundle, config, output_root, device, args):
    spec = dict(EXPERIMENTS[name])
    run_config = dict(config, **spec, experiment_name=name, seed=seed)
    checkpoint_dir = output_root / 'checkpoints' / name / 'seed_{}'.format(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_root / 'training_logs' / name).mkdir(parents=True, exist_ok=True)
    (output_root / 'metrics' / name).mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / 'checkpoint.pth'
    history_path = output_root / 'training_logs' / name / 'seed_{}.csv'.format(seed)
    metrics_path = output_root / 'metrics' / name / 'seed_{}.csv'.format(seed)
    prediction_dir = output_root / 'predictions' / name / 'seed_{}'.format(seed)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and experiment_complete(output_root, name, seed):
        print('Resume: {} seed {} is complete.'.format(name, seed))
        return
    set_seed(seed)
    model = build_model('xresnet1d101', 5, **spec, emd_features=len(bundle['emd_columns'])).to(device)
    parameter_count, trainable_count = count_parameters(model)
    print('Running {} seed {}: parameters={}'.format(name, seed, parameter_count))
    batch_size = config['batch_size']
    best_epoch = best_loss = training_seconds = None
    try:
        if not checkpoint.exists() and not args.skip_training and not args.evaluate_only:
            for candidate_bs in dict.fromkeys([batch_size, 64, 32]):
                if candidate_bs > batch_size:
                    continue
                try:
                    train_loader = build_loader(bundle['splits']['train'], spec['use_emd'], candidate_bs, True)
                    valid_loader = build_loader(bundle['splits']['val'], spec['use_emd'], candidate_bs)
                    run_config['actual_batch_size'] = candidate_bs
                    best_epoch, best_loss, training_seconds = train_model(
                        model, train_loader, valid_loader, run_config, device, checkpoint, history_path)
                    break
                except RuntimeError as error:
                    if 'out of memory' not in str(error).lower() or candidate_bs == 32:
                        raise
                    print('CUDA OOM at batch size {}; retrying smaller batch.'.format(candidate_bs))
                    del model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    model = build_model('xresnet1d101', 5, **spec, emd_features=len(bundle['emd_columns'])).to(device)
        if not checkpoint.exists():
            raise FileNotFoundError('No checkpoint for {} seed {}'.format(name, seed))
        state = load_checkpoint(model, checkpoint, device)
        best_epoch = state.get('epoch', best_epoch)
        best_loss = state.get('best_valid_loss', best_loss)
        val_loader = build_loader(bundle['splits']['val'], spec['use_emd'], config['batch_size'])
        val_prob, val_y, _ = predict(model, val_loader, device, spec['use_emd'])
        global_threshold, per_class_thresholds, threshold_curve = threshold_results(val_y, val_prob)
        json_dump(checkpoint_dir / 'thresholds.json', {'threshold_0.5': .5,
                  'best_global_threshold': global_threshold, 'per_class_thresholds': per_class_thresholds})
        pd.DataFrame(threshold_curve, columns=['threshold', 'validation_macro_f1']).to_csv(
            output_root / 'metrics' / name / 'seed_{}_threshold_search.csv'.format(seed), index=False)
        save_predictions(prediction_dir / 'validation_predictions.csv', bundle['splits']['val']['ids'], val_y,
                         val_prob, (val_prob >= np.array([per_class_thresholds[x] for x in CLASS_NAMES])).astype(int))
        all_overall, all_per_class = [], []
        scenarios = [('clean', None, bundle['splits']['test']['ecg'], bundle['splits']['test']['emd'],
                      'clean_original', 'clean', {'number_of_test_records': len(bundle['splits']['test']['ids']),
                      'number_of_matched_ecg_records': len(bundle['splits']['test']['ids']),
                      'number_of_matched_emd_records': len(bundle['splits']['test']['ids']),
                      'duplicate_record_ids': [], 'missing_record_ids': [], 'label_consistency': True})]
        for scenario, snr in SNR_SCENARIOS[1:]:
            ecg, check = load_noisy_ecg(bundle, scenario, snr)
            emd, source, feature_scenario, emd_check = scenario_emd(bundle, scenario)
            check.update(emd_check)
            scenarios.append((scenario, snr, ecg, emd, source, feature_scenario, check))
        for scenario, snr, ecg, emd, emd_source, feature_scenario, checks in scenarios:
            split = dict(ids=bundle['splits']['test']['ids'], ecg=ecg, emd=emd,
                         labels=bundle['splits']['test']['labels'])
            test_loader = build_loader(split, spec['use_emd'], config['batch_size'])
            probabilities, y_test, inference_ms = predict(model, test_loader, device, spec['use_emd'])
            json_dump(output_root / 'metrics' / name / 'seed_{}_{}_integrity.json'.format(seed, scenario), checks)
            for strategy, threshold in [('threshold_0.5', .5), ('best_global_threshold', global_threshold),
                                        ('per_class_thresholds', np.array([per_class_thresholds[x] for x in CLASS_NAMES]))]:
                prediction = (probabilities >= threshold).astype(int)
                metadata = dict(experiment_name=name, seed=seed, ecg_scenario=scenario,
                                feature_scenario=feature_scenario if spec['use_emd'] else 'not_used',
                                emd_source=emd_source if spec['use_emd'] else 'not_used',
                                threshold_strategy=strategy, threshold=json.dumps(threshold.tolist()) if hasattr(threshold, 'tolist') else threshold,
                                use_cbam=spec['use_cbam'], use_se=spec['use_se'], use_emd=spec['use_emd'],
                                se_reduction=config.get('se_reduction', 16), fusion_type=spec['fusion_type'],
                                best_epoch=best_epoch, best_valid_loss=best_loss,
                                training_time_seconds=training_seconds, parameter_count=parameter_count,
                                trainable_parameter_count=trainable_count, inference_time_per_sample_ms=inference_ms,
                                actual_batch_size=run_config.get('actual_batch_size', config['batch_size']),
                                checkpoint_path=str(checkpoint), status='completed')
                overall, per_class = metric_rows(y_test, probabilities, prediction, metadata)
                all_overall.append(overall)
                all_per_class.extend(per_class)
                if strategy == 'per_class_thresholds':
                    save_predictions(prediction_dir / 'test_predictions_{}.csv'.format(scenario), split['ids'],
                                     y_test, probabilities, prediction)
        pd.DataFrame(all_overall).to_csv(metrics_path, index=False)
        pd.DataFrame(all_per_class).to_csv(output_root / 'metrics' / name / 'seed_{}_per_class.csv'.format(seed), index=False)
        json_dump(checkpoint_dir / 'model_info.json', dict(experiment_name=name, seed=seed, **spec,
                  best_epoch=best_epoch, best_valid_loss=best_loss, training_time_seconds=training_seconds,
                  parameter_count=parameter_count, trainable_parameter_count=trainable_count,
                  actual_batch_size=run_config.get('actual_batch_size', config['batch_size']),
                  checkpoint_path=str(checkpoint), status='completed'))
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def make_figures(output_root, summary, per_class):
    figures = output_root / 'figures'
    figures.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(figures / (name + '.png'), dpi=150); plt.savefig(figures / (name + '.pdf')); plt.close()
    for name in summary.experiment_name.unique():
        history_files = list((output_root / 'training_logs' / name).glob('seed_*.csv')) if (output_root / 'training_logs' / name).exists() else []
        for history in history_files:
            frame = pd.read_csv(history); plt.figure(); plt.plot(frame.epoch, frame.train_loss, label='train'); plt.plot(frame.epoch, frame.valid_loss, label='valid'); plt.legend(); plt.xlabel('Epoch'); plt.ylabel('BCE loss'); plt.title(name); save('loss_' + name + '_' + history.stem)
    threshold = summary[(summary.ecg_scenario == 'clean') & (summary.threshold_strategy == 'per_class_thresholds')]
    for metric, title in [('macro_roc_auc', 'Clean Macro ROC-AUC'), ('macro_f1', 'Clean Macro F1')]:
        plt.figure(figsize=(9, 4)); plt.bar(threshold.experiment_name, threshold[metric]); plt.xticks(rotation=20, ha='right'); plt.ylabel(metric); plt.title(title); save('clean_' + metric)
    snr_order = ['snrm6', 'snr0', 'snr6', 'snr12', 'snr24']
    noisy = summary[(summary.ecg_scenario.isin(snr_order)) & (summary.threshold_strategy == 'per_class_thresholds')]
    for metric in ['macro_roc_auc', 'macro_f1']:
        plt.figure()
        for name, frame in noisy.groupby('experiment_name'):
            frame = frame.set_index('ecg_scenario').reindex(snr_order); plt.plot([-6, 0, 6, 12, 24], frame[metric], marker='o', label=name)
        plt.xlabel('SNR (dB)'); plt.ylabel(metric); plt.legend(fontsize=7); save(metric + '_vs_snr')
    clean = threshold.set_index('experiment_name')
    for metric in ['macro_roc_auc', 'macro_f1']:
        plt.figure()
        for name, frame in noisy.groupby('experiment_name'):
            base = clean.loc[name, metric] if name in clean.index else np.nan
            frame = frame.set_index('ecg_scenario').reindex(snr_order); plt.plot([-6, 0, 6, 12, 24], base - frame[metric], marker='o', label=name)
        plt.xlabel('SNR (dB)'); plt.ylabel(metric + ' drop from clean'); plt.legend(fontsize=7); save(metric + '_drop_vs_snr')
    pc = per_class[(per_class.ecg_scenario == 'clean') & (per_class.threshold_strategy == 'per_class_thresholds')]
    if len(pc):
        pivot = pc.pivot(index='class_name', columns='experiment_name', values='f1'); pivot.plot(kind='bar', figsize=(10, 4)); plt.ylabel('F1'); plt.title('Clean per-class F1'); save('per_class_f1')
    if len(threshold):
        plt.figure(); plt.scatter(threshold.parameter_count, threshold.macro_roc_auc)
        for _, row in threshold.iterrows(): plt.annotate(row.experiment_name, (row.parameter_count, row.macro_roc_auc), fontsize=7)
        plt.xlabel('Parameter count'); plt.ylabel('Clean macro ROC-AUC'); save('parameters_vs_macro_auc')
    for path in (output_root / 'metrics').glob('*threshold_search.csv'):
        frame = pd.read_csv(path); plt.figure(); plt.plot(frame.threshold, frame.validation_macro_f1); plt.xlabel('Threshold'); plt.ylabel('Validation macro F1'); plt.title(path.stem); save(path.stem)


def final_report(output_root):
    frames = [pd.read_csv(path) for path in (output_root / 'metrics').glob('*/*seed_*.csv') if 'per_class' not in path.name and 'threshold' not in path.name]
    per_class_frames = [pd.read_csv(path) for path in (output_root / 'metrics').glob('*/*per_class.csv')]
    if not frames:
        return
    summary = pd.concat(frames, ignore_index=True)
    completed_path = output_root / 'completed_experiments.json'
    if completed_path.exists():
        with open(completed_path) as file:
            completed = json.load(file)
        failed = []
        for key, status in completed.items():
            if status == 'failed':
                name, seed = key.rsplit('_', 1)
                failed.append({'experiment_name': name, 'seed': int(seed), 'status': 'failed'})
        if failed:
            summary = pd.concat([summary, pd.DataFrame(failed)], ignore_index=True, sort=False)
    per_class = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()
    report = output_root / 'final_report'
    summary.to_csv(report / 'ablation_summary.csv', index=False)
    selected = summary[summary.threshold_strategy == 'per_class_thresholds'].copy()
    clean = selected[selected.ecg_scenario == 'clean']; clean.to_csv(report / 'ablation_clean_comparison.csv', index=False)
    selected[selected.ecg_scenario != 'clean'].to_csv(report / 'ablation_snr_comparison.csv', index=False)
    rows = []
    comparisons = [('CBAM contribution', 'cbam_xresnet1d101', 'xresnet1d101_baseline'),
                   ('EMD late-fusion contribution', 'xresnet1d101_emd_late_fusion', 'xresnet1d101_baseline'),
                   ('EMD contribution with CBAM', 'cbam_xresnet1d101_emd_late_fusion', 'cbam_xresnet1d101'),
                   ('CBAM contribution with EMD', 'cbam_xresnet1d101_emd_late_fusion', 'xresnet1d101_emd_late_fusion'),
                   ('Complete model improvement', 'cbam_xresnet1d101_emd_late_fusion', 'xresnet1d101_baseline')]
    metrics = ['macro_roc_auc', 'macro_pr_auc', 'macro_f1', 'micro_f1', 'exact_match_accuracy']
    for scenario in ['clean', 'snr0', 'snrm6']:
        values = selected[selected.ecg_scenario == scenario].groupby('experiment_name')[metrics].mean()
        for label, newer, older in comparisons:
            if newer in values.index and older in values.index:
                row = {'comparison': label, 'scenario': scenario}; row.update((values.loc[newer] - values.loc[older]).to_dict()); rows.append(row)
    pd.DataFrame(rows).to_csv(report / 'ablation_contributions.csv', index=False)
    robust = []
    for name, frame in selected.groupby('experiment_name'):
        clean_row = frame[frame.ecg_scenario == 'clean'].iloc[0] if len(frame[frame.ecg_scenario == 'clean']) else None
        noisy = frame[frame.ecg_scenario != 'clean']
        if clean_row is not None and len(noisy):
            for _, row in noisy.iterrows():
                robust.append(dict(experiment_name=name, scenario=row.ecg_scenario,
                    auc_drop=clean_row.macro_roc_auc-row.macro_roc_auc, f1_drop=clean_row.macro_f1-row.macro_f1,
                    auc_retention=row.macro_roc_auc/clean_row.macro_roc_auc, f1_retention=row.macro_f1/clean_row.macro_f1))
    robust_frame = pd.DataFrame(robust); robust_frame.to_csv(report / 'robustness_metrics.csv', index=False)
    noisy_means = selected[selected.ecg_scenario != 'clean'].groupby('experiment_name')[['macro_roc_auc', 'macro_f1']].mean().rename(columns={'macro_roc_auc':'mean_noisy_macro_auc', 'macro_f1':'mean_noisy_macro_f1'})
    if len(robust_frame):
        noisy_means['mean_auc_drop'] = robust_frame.groupby('experiment_name').auc_drop.mean()
        noisy_means['mean_f1_drop'] = robust_frame.groupby('experiment_name').f1_drop.mean()
    noisy_means.reset_index().to_csv(report / 'mean_noisy_metrics.csv', index=False)
    aggregate_columns = selected.select_dtypes(include=[np.number]).columns.tolist()
    aggregate = selected.groupby(['experiment_name', 'ecg_scenario', 'threshold_strategy'])[aggregate_columns].agg(
        ['mean', 'std']).reset_index()
    aggregate.to_csv(report / 'ablation_seed_mean_std.csv', index=False)
    best = {'best_clean_model': clean.loc[clean.macro_roc_auc.idxmax(), 'experiment_name'],
            'best_mean_noisy_model': noisy_means.mean_noisy_macro_auc.idxmax(),
            'best_minus6db_model': selected[selected.ecg_scenario == 'snrm6'].sort_values('macro_roc_auc', ascending=False).iloc[0].experiment_name,
            'smallest_performance_drop_model': noisy_means.mean_auc_drop.idxmin() if 'mean_auc_drop' in noisy_means else None,
            'best_macro_f1_model': clean.loc[clean.macro_f1.idxmax(), 'experiment_name'],
            'best_macro_roc_auc_model': clean.loc[clean.macro_roc_auc.idxmax(), 'experiment_name']}
    json_dump(report / 'best_model_summary.json', best)
    make_figures(output_root, summary, per_class)
    console = clean.pivot_table(index='experiment_name', values=['macro_roc_auc', 'macro_f1'], aggfunc='mean')
    for scenario in ['snr0', 'snrm6']:
        console['{}_auc'.format(scenario)] = selected[selected.ecg_scenario == scenario].groupby('experiment_name').macro_roc_auc.mean()
    print(console.to_string())
    print('Result directory: {}'.format(output_root))


def main():
    args = parse_args(); config = read_config(args); output_root = choose_output_dir(config)
    prepare_directories(output_root); config['output_dir'] = str(output_root)
    json_dump(output_root / 'config' / 'resolved_config.json', config)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Output directory: {}'.format(output_root)); eprint_environment(device)
    bundle = load_smoke_data(config, output_root) if args.smoke_test else load_data(config, output_root)
    bundle['scaler_path'] = output_root / 'config' / 'standard_scaler.pkl'
    smoke_test(bundle, config, device)
    if args.smoke_test:
        return
    completed = {}
    for seed in config['seeds']:
        for name in config['experiments']:
            try:
                run_one(name, seed, bundle, config, output_root, device, args)
                completed['{}_{}'.format(name, seed)] = 'completed'
            except Exception:
                error_path = output_root / 'errors' / '{}_seed_{}.log'.format(name, seed)
                error_path.write_text(traceback.format_exc())
                completed['{}_{}'.format(name, seed)] = 'failed'
                print('FAILED {} seed {}; continuing. See {}'.format(name, seed, error_path))
            json_dump(output_root / 'completed_experiments.json', completed)
    final_report(output_root)


if __name__ == '__main__':
    main()
