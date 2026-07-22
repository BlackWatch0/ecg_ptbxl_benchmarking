import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from utils import data_assets


def test_load_metadata_parses_scp_codes(tmp_path):
    pd.DataFrame({'ecg_id': [1], 'scp_codes': ["{'NORM': 100}"], 'strat_fold': [10]}).to_csv(
        tmp_path / 'ptbxl_database_clean_no_noise.csv', index=False)
    metadata = data_assets.load_metadata(tmp_path)
    assert metadata.loc[1, 'scp_codes'] == {'NORM': 100}


def test_resolve_emd_paths_prefers_active_layout(tmp_path):
    active = tmp_path / 'emd_features' / 'ptbxl_original_database_plus_mixed' / 'mixed_snr6'
    active.mkdir(parents=True)
    path = active / 'mixed_snr6_plus_mixed_EMD_Features_reduced_features.csv'
    path.touch()
    assert data_assets.resolve_emd_paths(tmp_path)['snr6'] == path


def test_load_noisy_manifest_resolves_prepared_root(tmp_path):
    root = tmp_path / 'ptbxl_noisy_mixed_shared'
    root.mkdir()
    pd.DataFrame({'ecg_id': [1], 'snr_target_db': [6], 'wfdb_record_relative': ['x']}).to_csv(
        root / 'ptbxl_noisy_mixed_shared_manifest.csv', index=False)
    manifest, resolved = data_assets.load_noisy_manifest(tmp_path)
    assert resolved == root
    assert manifest.ecg_id.tolist() == [1]
