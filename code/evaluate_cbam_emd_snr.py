import ast
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from configs.cbam_configs import conf_cbam_xresnet1d101_late_fusion_superdiagnostic
from models.cbam_xresnet1d_model import cbam_xresnet1d_model
from utils import utils
from utils.emd_features import apply_emd_standardizer, load_emd_features


DATA_ROOT = Path('../data')
OUTPUT_ROOT = Path('../output')
EXPERIMENT_NAME = 'exp_emd_late_fusion_superdiagnostic'
MODEL_NAME = 'cbam_xresnet1d101_late_fusion_superdiagnostic'
NOISY_ROOT = DATA_ROOT / 'ptbxl_noisy_mixed_shared'
SNR_SCENARIOS = [('snr24', 24), ('snr12', 12), ('snr6', 6), ('snr0', 0), ('snrm6', -6)]


def safe_auc(y_true, y_prob, metric):
    if len(np.unique(y_true)) < 2:
        return np.nan
    if metric == 'roc':
        return roc_auc_score(y_true, y_prob)
    return average_precision_score(y_true, y_prob)


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    roc = np.array([safe_auc(y_true[:, index], y_prob[:, index], 'roc') for index in range(y_true.shape[1])])
    pr = np.array([safe_auc(y_true[:, index], y_prob[:, index], 'pr') for index in range(y_true.shape[1])])
    return {
        'macro_roc_auc': np.nanmean(roc),
        'micro_roc_auc': roc_auc_score(y_true.ravel(), y_prob.ravel()),
        'macro_pr_auc': np.nanmean(pr),
        'micro_pr_auc': average_precision_score(y_true.ravel(), y_prob.ravel()),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'predicted_positive_rate': y_pred.mean(),
        'mean_predicted_labels': y_pred.sum(axis=1).mean(),
        'all_zero_prediction_rate': (y_pred.sum(axis=1) == 0).mean(),
        'valid_auc_classes': int(np.isfinite(roc).sum()),
    }


def load_noisy_waveforms(manifest, snr, record_ids):
    rows = manifest[(manifest.snr_target_db == snr) & manifest.ecg_id.isin(record_ids)].set_index('ecg_id')
    rows = rows.loc[record_ids]
    return np.array([wfdb.rdsamp(str(NOISY_ROOT / path))[0] for path in rows.wfdb_record_relative])


def load_scalers(data_root):
    with open(data_root / 'standard_scaler.pkl', 'rb') as file:
        raw_scaler = pickle.load(file)
    emd_scaler = np.load(data_root / 'emd_scaler.npz')
    return raw_scaler, emd_scaler['mean'], emd_scaler['std'], emd_scaler['feature_columns'].tolist()


def main():
    config = deepcopy(conf_cbam_xresnet1d101_late_fusion_superdiagnostic)
    model_root = OUTPUT_ROOT / EXPERIMENT_NAME / 'models' / MODEL_NAME
    data_root = OUTPUT_ROOT / EXPERIMENT_NAME / 'data'
    checkpoint = model_root / 'models' / '{}.pth'.format(MODEL_NAME)
    checkpoint_name = MODEL_NAME
    if not checkpoint.exists():
        checkpoint = model_root / 'models' / 'best_valid_loss.pth'
        checkpoint_name = 'best_valid_loss'
    if not checkpoint.exists():
        raise FileNotFoundError('Missing trained CBAM checkpoint: {}'.format(checkpoint))
    if config['parameters']['input_size'] != 10.0:
        raise ValueError('SNR evaluation requires the new 1000-point CBAM checkpoint')

    metadata = pd.read_csv(DATA_ROOT / 'ptbxl_clean_no_noise' / 'ptbxl_database_clean_no_noise.csv', index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    labels = utils.compute_label_aggregations(metadata, str(DATA_ROOT / 'ptbxl_clean_no_noise') + '/', 'superdiagnostic')
    _, labels, y, mlb = utils.select_data(
        np.empty(len(labels), dtype=object), labels, 'superdiagnostic', 0, '/tmp/',
        class_order=['NORM', 'MI', 'STTC', 'CD', 'HYP']
    )
    test_ids = labels.index[labels.strat_fold == 10]
    y_test = y[labels.strat_fold == 10]
    raw_scaler, emd_mean, emd_std, feature_columns = load_scalers(data_root)
    manifest = pd.read_csv(NOISY_ROOT / 'ptbxl_noisy_mixed_shared_manifest.csv')
    model_params = {key: value for key, value in config['parameters'].items() if key not in {
        'emd_feature_paths', 'emd_scenario', 'waveform_scenario', 'missing_record_policy',
        'feature_log_transform', 'log_feature_columns'
    }}
    model = cbam_xresnet1d_model(MODEL_NAME, y_test.shape[1], 100, model_root,
                                 (1000, 12), **model_params)
    model.name = checkpoint_name
    results = []
    for scenario, snr in SNR_SCENARIOS:
        print('Evaluating {} dB...'.format(snr))
        X_ecg = utils.apply_standardizer(load_noisy_waveforms(manifest, snr, test_ids), raw_scaler)
        emd_ids, X_emd, incomplete = load_emd_features(
            config['parameters']['emd_feature_paths'][scenario], labels, feature_columns,
            test_ids, missing_record_policy='error'
        )
        if incomplete or not np.array_equal(emd_ids, test_ids):
            raise ValueError('SNR {} ECG and EMD record IDs are not aligned'.format(snr))
        X_emd = apply_emd_standardizer(X_emd, emd_mean, emd_std)
        paired = list(zip(X_ecg, X_emd))
        y_prob = model.predict(paired)
        y_logits = model.predict_logits(paired)
        scenario_root = model_root / 'snr_{}'.format(snr)
        scenario_root.mkdir(parents=True, exist_ok=True)
        np.save(scenario_root / 'y_test_prob.npy', y_prob)
        np.save(scenario_root / 'y_test_logits.npy', y_logits)
        for threshold in [0.5, 0.3]:
            results.append({'scenario': scenario, 'target_snr_db': snr, 'threshold': threshold,
                            'n_test_records': len(test_ids), **evaluate(y_test, y_prob, threshold)})
    pd.DataFrame(results).to_csv(model_root / 'snr_test_results.csv', index=False)
    with open(model_root / 'snr_test_config.json', 'w') as file:
        json.dump({'input_size': 1000, 'sampling_rate': 100, 'test_fold': 10,
                   'feature_columns': feature_columns, 'class_names': mlb.classes_.tolist(),
                   'scenarios': SNR_SCENARIOS}, file, indent=2)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
