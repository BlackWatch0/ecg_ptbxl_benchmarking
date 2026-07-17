"""Read-only diagnostics for CBAM-xResNet1D + EMD late fusion."""
import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from models.cbam_xresnet1d import build_model
from run_ablation_study import (CLASS_NAMES, EXPERIMENTS, build_loader, load_data,
                                 load_smoke_data, read_config, safe_metric)


def write_json(path, value):
    with open(path, 'w') as handle:
        json.dump(value, handle, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def score(labels, probabilities):
    prediction = probabilities >= .5
    per_class = [safe_metric(roc_auc_score, labels[:, i], probabilities[:, i]) for i in range(5)]
    return {'macro_roc_auc': float(np.nanmean(per_class)),
            'macro_f1': float(f1_score(labels, prediction, average='macro', zero_division=0)),
            'prediction_rate': float(prediction.mean()), 'per_class_roc_auc': per_class}


def collect(model, loader, device, action=None, gradients=False):
    """Bound analysis to 20 batches so complete diagnostics remain low overhead."""
    model.eval(); probabilities = []; labels = []; vectors = {}; gradient = []
    for index, (ecg, emd, target) in enumerate(loader):
        if index == 20:
            break
        ecg, emd = ecg.to(device), emd.to(device)
        if action == 'zero':
            emd = torch.zeros_like(emd)
        elif action == 'shuffle':
            emd = emd[torch.randperm(len(emd), device=device)]
        if gradients:
            emd.requires_grad_(True)
        logits, intermediates = model(ecg, emd, return_intermediates=True)
        if gradients:
            model.zero_grad(set_to_none=True); logits.sum().backward()
            gradient.append(emd.grad.detach().abs().mean((0, 1)).cpu().numpy())
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(target.numpy())
        for name, value in intermediates.items():
            vectors.setdefault(name, []).append(value.detach().cpu().numpy())
    return (np.vstack(labels), np.vstack(probabilities),
            {name: np.vstack(value) for name, value in vectors.items()},
            np.mean(gradient, axis=0) if gradient else None)


def history_report(source, figures):
    result = []
    for path in source.glob('training_logs/*/seed_*.csv'):
        try:
            frame = pd.read_csv(path)
            if {'epoch', 'train_loss', 'valid_loss'} <= set(frame.columns):
                result.append({'model': path.parent.name, 'seed': path.stem, 'epochs': len(frame),
                               'best_epoch': int(frame.loc[frame.valid_loss.idxmin(), 'epoch']),
                               'best_valid_loss': float(frame.valid_loss.min())})
                plt.plot(frame.epoch, frame.valid_loss, label=path.parent.name)
        except Exception as error:
            result.append({'file': str(path), 'status': 'unavailable', 'reason': str(error)})
    if result:
        plt.xlabel('epoch'); plt.ylabel('validation BCE'); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(figures / 'validation_loss_comparison.png', dpi=160); plt.close()
    return result or [{'status': 'unavailable', 'reason': 'No readable training histories'}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation-root', required=True)
    parser.add_argument('--config', default='../configs/ablation_cbam_emd.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--device')
    args = parser.parse_args()
    source = Path(args.ablation_root).resolve()
    output = Path(args.output_dir or source.parent / ('cbam_emd_diagnostics_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    figures = output / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    report = {'scope': 'read_only', 'source_ablation_root': str(source), 'device': str(device),
              'module_mapping': {'ecg_encoder': 'ecg_backbone', 'emd_encoder': 'feature_encoder',
                                 'fusion_head': 'fusion', 'classifier': 'output_layer',
                                 'cbam': 'ecg_backbone.*.cbam'}, 'unavailable': []}
    try:
        config_args = type('ConfigArgs', (), {'config': args.config, 'experiments': None, 'seeds': None, 'output_dir': None})()
        config = read_config(config_args)
        bundle = load_smoke_data(config, output) if args.smoke_test else load_data(config, output)
        report['resolved_config'] = config
        np.savez(output / 'emd_normalization.npz', mean=bundle['emd_mean'], std=bundle['emd_std'],
                 feature_columns=np.array(bundle['emd_columns']))
        report['emd_normalization'] = {'fitted_on': 'train only', 'feature_columns': bundle['emd_columns'],
            'standardized_train_mean': bundle['splits']['train']['emd'].mean((0, 1)).tolist(),
            'standardized_train_std': bundle['splits']['train']['emd'].std((0, 1)).tolist()}
        checkpoint = source / 'checkpoints/cbam_xresnet1d101_emd_late_fusion' / ('seed_{}'.format(args.seed)) / 'checkpoint.pth'
        if not checkpoint.exists():
            raise FileNotFoundError('Expected checkpoint is absent: {}'.format(checkpoint))
        model = build_model('xresnet1d101', 5, **EXPERIMENTS['cbam_xresnet1d101_emd_late_fusion'],
                            emd_features=len(bundle['emd_columns'])).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(state['model'])
        labels, probabilities, vectors, gradient = collect(model, build_loader(bundle['splits']['test'], True, 32), device, gradients=True)
        report['checkpoint'] = {'path': str(checkpoint), 'epoch': state.get('epoch'), 'best_valid_loss': state.get('best_valid_loss')}
        report['test_metrics'] = score(labels, probabilities)
        report['activation_norms'] = {name: {'mean': float(value.mean()), 'std': float(value.std()),
                                      'mean_l2': float(np.linalg.norm(value, axis=1).mean()), 'shape': list(value.shape)}
                                      for name, value in vectors.items()}
        report['gradient_mean_abs_by_emd_feature'] = dict(zip(bundle['emd_columns'], gradient.tolist()))
        loader = build_loader(bundle['splits']['test'], True, 32)
        report['branch_ablation'] = {name: score(labels, collect(model, loader, device, action=name)[1]) for name in ('zero', 'shuffle')}
        report['branch_ablation']['ecg_only'] = {'status': 'unavailable', 'reason': 'Requires the separately trained ECG-only checkpoint; no retraining is performed.'}
        raw = bundle['splits']['test'].get('emd_raw', bundle['splits']['test']['emd'])
        rows = [{'feature': feature, 'class': label, 'pearson_r': float(np.corrcoef(raw[:, :, fi].mean(1), labels[:, ci])[0, 1])}
                for fi, feature in enumerate(bundle['emd_columns']) for ci, label in enumerate(CLASS_NAMES)]
        pd.DataFrame(rows).to_csv(output / 'emd_feature_label_associations.csv', index=False)
        report['history_comparison'] = history_report(source, figures)
    except Exception as error:
        report['unavailable'].append({'status': 'unavailable', 'reason': repr(error)})
    write_json(output / 'diagnostic_summary.json', report)
    shutil.make_archive(str(output), 'zip', output)
    print(json.dumps({'output': str(output), 'bundle': str(output) + '.zip', 'unavailable': report['unavailable']}, indent=2))


if __name__ == '__main__':
    main()
