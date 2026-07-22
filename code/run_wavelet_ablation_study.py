import argparse
import ast
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.wavelet_feature_store import WaveletFeatureStore
from models.wavelet_late_fusion import build_wavelet_ablation_model
from utils import utils


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SNR_LEVELS = [24, 12, 6, 0, -6]
EXPERIMENTS = {
    'xresnet1d101_baseline': (False, False),
    'cbam_xresnet1d101': (True, False),
    'xresnet1d101_wavelet_late_fusion': (False, True),
    'cbam_xresnet1d101_wavelet_late_fusion': (True, True),
}


class ECGWaveletDataset(Dataset):
    def __init__(self, signals, labels, features=None):
        self.signals, self.labels, self.features = signals, labels, features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ecg = torch.from_numpy(np.asarray(self.signals[index], dtype=np.float32).T)
        label = torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32))
        if self.features is None:
            return ecg, label
        return ecg, torch.from_numpy(self.features[index].astype(np.float32)), label


def parse_args():
    parser = argparse.ArgumentParser(description='Standalone PTB-XL Wavelet late-fusion ablation runner.')
    parser.add_argument('--config', default=str(REPO_ROOT / 'configs' / 'ablation_cbam_wavelet.yaml'))
    parser.add_argument('--output-dir')
    parser.add_argument('--device')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()


def read_config(args):
    with open(args.config) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get('task') != 'superdiagnostic' or config.get('class_names') != CLASS_NAMES:
        raise ValueError('This runner requires the canonical five superdiagnostic classes.')
    config['experiments'] = args.experiments or config.get('experiments', list(EXPERIMENTS))
    if set(config['experiments']).difference(EXPERIMENTS):
        raise ValueError('Unknown experiment requested.')
    return config


def path_from_config(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_extract(archive, destination, archive_type):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if archive_type == 'zip':
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            for name in names:
                if not is_within(root, root / name):
                    raise ValueError('Unsafe zip member: {}'.format(name))
            handle.extractall(destination)
    else:
        mode = 'r:gz' if archive_type == 'tar.gz' else 'r:'
        with tarfile.open(archive, mode) as handle:
            for member in handle.getmembers():
                if member.issym() or member.islnk() or not is_within(root, root / member.name):
                    raise ValueError('Unsafe tar member: {}'.format(member.name))
            handle.extractall(destination)


def is_within(root, candidate):
    try:
        candidate.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def prepare_archives(config):
    sources = config['data_sources']
    roots = {}
    for key in ('ptbxl_clean', 'wavelet', 'mixed_noisy_ecg', 'denoised_ecg'):
        item = sources[key]
        mounted, runtime, destination = map(path_from_config, (item['mounted_archive'], item['runtime_archive'], item['extract_dir']))
        if not runtime.exists():
            runtime.parent.mkdir(parents=True, exist_ok=True)
            if mounted.exists():
                if sources.get('copy_archives_to_runtime', True):
                    shutil.copy2(mounted, runtime)
                else:
                    runtime = mounted
            elif sources.get('fetch_if_mounted_missing') and item.get('fetch_url'):
                urlretrieve(item['fetch_url'], runtime)
            else:
                raise FileNotFoundError('Required archive is unavailable: {}'.format(mounted))
        if not destination.exists() or not any(destination.iterdir()):
            safe_extract(runtime, destination, item['archive_type'])
        roots[key] = destination
    return roots


def find_root(extract_dir, required_name, fallback=None):
    candidates = [extract_dir]
    candidates.extend(path for path in extract_dir.rglob(required_name) if path.is_file())
    if fallback is not None:
        candidates.extend(path for path in fallback.rglob(required_name) if path.is_file())
    for candidate in candidates:
        parent = candidate.parent if candidate.is_file() else candidate
        if (parent / required_name).exists():
            return parent
    raise FileNotFoundError('Could not resolve root containing {}'.format(required_name))


def canonical_id(value):
    text = str(value).replace('\\', '/')
    for token in reversed(text.replace('.', '_').split('_')):
        if token.isdigit():
            return int(token)
    import re
    matches = re.findall(r'(?<!\d)(\d{1,6})(?!\d)', text)
    if not matches:
        raise ValueError('Cannot derive canonical ECG ID from {}'.format(value))
    return int(matches[-1])


def record_map(root):
    result = {}
    for header in root.rglob('*.hea'):
        record = header.with_suffix('')
        try:
            record_id = canonical_id(record.name)
        except ValueError:
            continue
        if record_id in result:
            raise ValueError('Duplicate canonical ECG ID {} in {}'.format(record_id, root))
        result[record_id] = record
    return result


def discover_scenarios(noisy_root, denoised_root, ids):
    scenarios = {'clean': None}
    for family, root in [('noisy', noisy_root), ('denoised', denoised_root)]:
        headers = list(root.rglob('*.hea'))
        for snr in SNR_LEVELS:
            patterns = ['snr{}'.format(snr), 'snr_{}'.format(snr), '{}db'.format(snr)]
            if snr < 0:
                patterns.extend(['snrm{}'.format(abs(snr)), 'minus{}'.format(abs(snr))])
            selected = [header for header in headers if any(pattern in str(header).lower() for pattern in patterns)]
            mapping = {}
            for header in selected:
                identifier = canonical_id(header.stem)
                if identifier in mapping:
                    raise ValueError('Duplicate {} SNR {} record {}'.format(family, snr, identifier))
                mapping[identifier] = header.with_suffix('')
            missing = sorted(set(map(int, ids)).difference(mapping))
            if missing:
                raise ValueError('Missing {} SNR {} records, first IDs: {}'.format(family, snr, missing[:10]))
            scenarios['{}_snr{}'.format(family, 'm{}'.format(abs(snr)) if snr < 0 else snr)] = mapping
    if len(scenarios) != 11:
        raise ValueError('Expected clean plus ten noisy/denoised scenarios.')
    return scenarios


def load_records(paths):
    records = [wfdb.rdsamp(str(path))[0].astype(np.float32) for path in paths]
    if any(record.shape != (1000, 12) for record in records):
        raise ValueError('All WFDB records must have shape (1000, 12).')
    return np.asarray(records, dtype=np.float32)


def load_bundle(roots, config, output_root, rebuild):
    clean_root = find_root(roots['ptbxl_clean'], 'ptbxl_database.csv')
    statements = clean_root / 'scp_statements.csv'
    records100 = clean_root / 'records100'
    if not statements.is_file() or not records100.is_dir():
        raise FileNotFoundError('Clean PTB-XL root must contain scp_statements.csv and records100: {}'.format(clean_root))
    hea_count = sum(1 for _ in records100.rglob('*.hea'))
    dat_count = sum(1 for _ in records100.rglob('*.dat'))
    if hea_count != 21799 or dat_count != 21799 or hea_count != dat_count:
        raise ValueError('PTB-XL records100 audit failed: hea={}, dat={}'.format(hea_count, dat_count))
    metadata = pd.read_csv(clean_root / 'ptbxl_database.csv', index_col='ecg_id')
    if len(metadata) != 21799 or not {'filename_lr', 'scp_codes', 'strat_fold'}.issubset(metadata.columns):
        raise ValueError('PTB-XL metadata audit failed')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    labels = utils.compute_label_aggregations(metadata.copy(), str(clean_root) + '/', 'superdiagnostic')
    _, labels, targets, mlb = utils.select_data(np.empty(len(labels), dtype=object), labels, 'superdiagnostic', 0,
                                                 str(output_root / 'config') + '/', CLASS_NAMES)
    if mlb.classes_.tolist() != CLASS_NAMES or len(labels) != len(targets):
        raise ValueError('PTB label audit failed: class order or row count differs.')
    ids = labels.index.to_numpy(dtype=np.int64)
    clean_map = record_map(roots['ptbxl_clean'])
    if clean_root != roots['ptbxl_clean']:
        for record_id, record_path in record_map(clean_root).items():
            if record_id in clean_map and clean_map[record_id] != record_path:
                raise ValueError('Duplicate clean ECG ID {} across archive and metadata roots'.format(record_id))
            clean_map[record_id] = record_path
    missing = sorted(set(ids).difference(clean_map))
    if missing:
        raise ValueError('Clean PTB audit missing records, first IDs: {}'.format(missing[:10]))
    masks = {'train': labels.strat_fold.to_numpy() <= 8, 'val': labels.strat_fold.to_numpy() == 9,
             'test': labels.strat_fold.to_numpy() == 10}
    splits = {}
    for name, mask in masks.items():
        split_ids = ids[mask]
        splits[name] = {'ids': split_ids, 'labels': targets[mask].astype(np.float32),
                        'raw_ecg': load_records([clean_map[int(value)] for value in split_ids])}
    if set(splits['train']['ids']) & set(splits['val']['ids']) or set(splits['train']['ids']) & set(splits['test']['ids']) or set(splits['val']['ids']) & set(splits['test']['ids']):
        raise ValueError('PTB fold audit found split overlap.')
    scaler = StandardScaler().fit(splits['train']['raw_ecg'].reshape(-1, 12))
    for split in splits.values():
        split['ecg'] = scaler.transform(split['raw_ecg'].reshape(-1, 12)).reshape(split['raw_ecg'].shape).astype(np.float32)
    cache_root = path_from_config(config['runtime_paths']['wavelet_cache'])
    store = WaveletFeatureStore(roots['wavelet'], cache_dir=cache_root, strict=True)
    for name in ('train', 'val', 'test'):
        canonical = ['{:05d}_lr'.format(value) for value in splits[name]['ids']]
        features, report = store.get_features(canonical, 'Denoised_Original_Signal')
        if len(features) != len(canonical):
            raise ValueError('Clean Wavelet ID alignment failed for {}'.format(name))
        splits[name]['features_raw'] = features
        store.cache('clean_original_{}'.format(name), canonical, features, report)
    feature_scaler = StandardScaler().fit(splits['train']['features_raw'].reshape(len(splits['train']['ids']), -1))
    for split in splits.values():
        split['features'] = feature_scaler.transform(split['features_raw'].reshape(len(split['ids']), -1)).reshape(split['features_raw'].shape).astype(np.float32)
    noisy_root = find_root(roots['mixed_noisy_ecg'], '*.hea') if False else roots['mixed_noisy_ecg']
    scenarios = discover_scenarios(noisy_root, roots['denoised_ecg'], splits['test']['ids'])
    audit = {'clean_root': str(clean_root), 'metadata_rows': len(metadata), 'records100_hea': hea_count,
             'records100_dat': dat_count, 'counts': {name: len(value['ids']) for name, value in splits.items()},
             'classes': mlb.classes_.tolist(), 'scenario_count': len(scenarios), 'wavelet_shape': [12, 6]}
    with open(output_root / 'config' / 'ptb_audit.json', 'w') as handle:
        json.dump(audit, handle, indent=2)
    return splits, scaler, feature_scaler, scenarios, store


def loader(split, use_wavelet, batch_size, shuffle=False):
    return DataLoader(ECGWaveletDataset(split['ecg'], split['labels'], split['features'] if use_wavelet else None),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def forward(model, batch, device, use_wavelet):
    if use_wavelet:
        ecg, features, labels = batch
        return model(ecg.to(device), features.to(device)), labels.to(device)
    ecg, labels = batch
    return model(ecg.to(device)), labels.to(device)


def predict(model, data_loader, device, use_wavelet):
    model.eval(); probabilities = []; labels = []
    with torch.no_grad():
        for batch in data_loader:
            logits, target = forward(model, batch, device, use_wavelet)
            probabilities.append(torch.sigmoid(logits).cpu().numpy()); labels.append(target.cpu().numpy())
    return np.vstack(probabilities), np.vstack(labels)


def train(model, train_loader, val_loader, config, device, checkpoint, resume, use_wavelet):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], epochs=config['epochs'], steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get('mixed_precision') and device.type == 'cuda'))
    start, best, stale = 0, float('inf'), 0
    last = checkpoint.with_name('last.pth')
    if resume and last.exists():
        state = load_torch(last, device); model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['amp']); start, best, stale = state['epoch'], state['best'], state['stale']
    for epoch in range(start, config['epochs']):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits, labels = forward(model, batch, device, use_wavelet); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
        model.eval(); losses = []
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = forward(model, batch, device, use_wavelet); losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).cpu()))
        valid = float(np.mean(losses)); improved = valid < best
        if improved:
            best, stale = valid, 0; torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'valid_loss': valid}, checkpoint)
        else:
            stale += 1
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'amp': scaler.state_dict(), 'epoch': epoch + 1, 'best': best, 'stale': stale}, last)
        print('epoch {}/{} valid_loss={:.6f}'.format(epoch + 1, config['epochs'], valid))
        if stale >= config.get('early_stopping_patience', 8):
            break
    state = load_torch(checkpoint, device); model.load_state_dict(state['model'])


def thresholds(y, probabilities):
    values = np.round(np.arange(.10, .91, .01), 2); per_class = []
    curve = [(float(value), float(f1_score(y, probabilities >= value, average='macro', zero_division=0))) for value in values]
    global_threshold = max(curve, key=lambda item: item[1])[0]
    for index in range(5):
        per_class.append(float(max(values, key=lambda value: f1_score(y[:, index], probabilities[:, index] >= value, zero_division=0))))
    return global_threshold, np.array(per_class, dtype=np.float32), curve


def metrics(y, probabilities, prediction):
    auc = [roc_auc_score(y[:, i], probabilities[:, i]) if len(np.unique(y[:, i])) == 2 else np.nan for i in range(5)]
    pr = [average_precision_score(y[:, i], probabilities[:, i]) if len(np.unique(y[:, i])) == 2 else np.nan for i in range(5)]
    return {'macro_roc_auc': float(np.nanmean(auc)), 'macro_pr_auc': float(np.nanmean(pr)),
            'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(y, prediction, average='micro', zero_division=0))}


def main():
    args = parse_args(); config = read_config(args)
    output = path_from_config(args.output_dir or config['runtime_paths'].get('output_dir', config.get('output_dir')))
    if str(output).startswith('/content/drive') and not Path('/content/drive/MyDrive').exists(): output = REPO_ROOT / 'results' / 'ablation_wavelet'
    for name in ('config', 'checkpoints', 'metrics', 'predictions'): (output / name).mkdir(parents=True, exist_ok=True)
    roots = prepare_archives(config)
    with open(output / 'config' / 'resolved_config.json', 'w') as handle:
        json.dump(dict(config, resolved_roots={name: str(path.resolve()) for name, path in roots.items()}), handle, indent=2)
    splits, ecg_scaler, feature_scaler, scenarios, store = load_bundle(roots, config, output, args.rebuild_cache)
    np.savez(output / 'config' / 'scalers.npz', ecg_mean=ecg_scaler.mean_, ecg_scale=ecg_scaler.scale_, wavelet_mean=feature_scaler.mean_, wavelet_scale=feature_scaler.scale_)
    if args.audit_only: return
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.smoke_test:
        for name, (cbam, wavelet) in EXPERIMENTS.items():
            model = build_wavelet_ablation_model(5, cbam, 'wavelet' if wavelet else None, 'concat' if wavelet else 'none', (12, 6) if wavelet else None).to(device)
            batch = next(iter(loader({key: value[:1] if isinstance(value, np.ndarray) else value for key, value in splits['train'].items() if key in ('ecg', 'labels', 'features')}, wavelet, 1)))
            logits, labels = forward(model, batch, device, wavelet)
            if logits.shape != (1, 5) or not torch.isfinite(logits).all(): raise ValueError('Smoke test failed: {}'.format(name))
            loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if not torch.isfinite(loss): raise ValueError('Smoke test loss failed: {}'.format(name))
        print('Smoke test passed.'); return
    rows = []
    for seed in config.get('seeds', [42]):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        for name in config['experiments']:
            cbam, wavelet = EXPERIMENTS[name]; run_dir = output / 'checkpoints' / name / 'seed_{}'.format(seed); run_dir.mkdir(parents=True, exist_ok=True); checkpoint = run_dir / 'best.pth'
            model = build_wavelet_ablation_model(5, cbam, 'wavelet' if wavelet else None, 'concat' if wavelet else 'none', (12, 6) if wavelet else None).to(device)
            if not checkpoint.exists() or args.resume: train(model, loader(splits['train'], wavelet, config['batch_size'], True), loader(splits['val'], wavelet, config['batch_size']), config, device, checkpoint, args.resume, wavelet)
            else: model.load_state_dict(load_torch(checkpoint, device)['model'])
            val_prob, val_y = predict(model, loader(splits['val'], wavelet, config['batch_size']), device, wavelet)
            global_threshold, per_class_thresholds, threshold_curve = thresholds(val_y, val_prob)
            (output / 'thresholds' / name).mkdir(parents=True, exist_ok=True)
            with open(output / 'thresholds' / name / 'seed_{}.json'.format(seed), 'w') as handle:
                json.dump({'threshold_0.5': 0.5, 'best_global_threshold': global_threshold,
                           'per_class_thresholds': dict(zip(CLASS_NAMES, per_class_thresholds.tolist())),
                           'selected_on': 'clean validation fold 9'}, handle, indent=2)
            pd.DataFrame(threshold_curve, columns=['threshold', 'validation_macro_f1']).to_csv(
                output / 'metrics' / '{}_seed_{}_threshold_search.csv'.format(name, seed), index=False)
            for scenario, mapping in scenarios.items():
                test = dict(splits['test'])
                if mapping is not None:
                    raw = load_records([mapping[int(value)] for value in test['ids']]); test['ecg'] = ecg_scaler.transform(raw.reshape(-1, 12)).reshape(raw.shape).astype(np.float32)
                    family, token = scenario.rsplit('_snr', 1)
                    snr = -int(token[1:]) if token.startswith('m') else int(token)
                    section = 'Denoised_Mixed_Noisy' if family == 'noisy' else 'Denoised_Denoised_Noisy'
                    canonical = ['{:05d}_lr'.format(value) for value in test['ids']]
                    features, report = store.get_features(canonical, section, snr)
                    if len(features) != len(canonical):
                        raise ValueError('Wavelet scenario alignment failed for {}'.format(scenario))
                    test['features'] = feature_scaler.transform(features.reshape(len(features), -1)).reshape(features.shape).astype(np.float32)
                    store.cache('{}_{}'.format(family, snr), canonical, features, report)
                probability, target = predict(model, loader(test, wavelet, config['batch_size']), device, wavelet)
                for strategy, threshold in [('threshold_0.5', .5), ('best_global_threshold', global_threshold), ('per_class_thresholds', per_class_thresholds)]:
                    prediction = (probability >= threshold).astype(int)
                    row = dict(experiment=name, seed=seed, scenario=scenario, use_cbam=cbam, use_wavelet=wavelet,
                               threshold_strategy=strategy, threshold=json.dumps(threshold.tolist()) if hasattr(threshold, 'tolist') else threshold)
                    row.update(metrics(target, probability, prediction)); rows.append(row)
                    frame = pd.DataFrame({'record_id': ['{:05d}_lr'.format(value) for value in test['ids']]})
                    for index, label in enumerate(CLASS_NAMES):
                        frame['true_' + label] = target[:, index].astype(int); frame['prob_' + label] = probability[:, index]; frame['pred_' + label] = prediction[:, index]
                    frame.to_csv(output / 'predictions' / '{}_seed_{}_{}_{}.csv'.format(name, seed, scenario, strategy), index=False)
    pd.DataFrame(rows).to_csv(output / 'metrics' / 'scenario_metrics.csv', index=False)
    pd.DataFrame(rows).groupby(['experiment', 'scenario', 'threshold_strategy'], as_index=False).mean(numeric_only=True).to_csv(
        output / 'metrics' / 'scenario_metrics_summary.csv', index=False)


if __name__ == '__main__':
    main()
