import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from utils.emd_features import CANDIDATE_EMD_FEATURES, LEAD_ORDER, load_emd_features, resolve_emd_feature_columns


def _rows(record_ids, missing=None):
    rows = []
    for record_id in record_ids:
        for lead_index, lead in enumerate(LEAD_ORDER, 1):
            if (record_id, lead_index) == missing:
                continue
            row = {'RecordNumber': record_id, 'LeadIndex': lead_index, 'Lead': lead, 'ProcessingStatus': 'Success'}
            row.update({column: record_id * 100 + lead_index for column in CANDIDATE_EMD_FEATURES})
            rows.append(row)
    return rows


def test_load_emd_features_sorts_and_aligns(tmp_path):
    path = tmp_path / 'emd.csv'
    pd.DataFrame(_rows([2, 1])).sample(frac=1, random_state=1).to_csv(path, index=False)
    metadata = pd.DataFrame({'ecg_id': [2, 1]}).set_index('ecg_id')
    columns = resolve_emd_feature_columns([path])
    record_ids, features, incomplete = load_emd_features(path, metadata, columns, np.array([2, 1]))
    assert record_ids.tolist() == [2, 1]
    assert features.shape == (2, 12, len(columns))
    assert features.dtype == np.float32
    assert features[0, 0, 0] == 201
    assert incomplete == []


def test_load_emd_features_drops_or_rejects_incomplete_records(tmp_path):
    path = tmp_path / 'emd.csv'
    pd.DataFrame(_rows([1, 2], missing=(2, 12))).to_csv(path, index=False)
    metadata = pd.DataFrame({'ecg_id': [1, 2]}).set_index('ecg_id')
    record_ids, features, incomplete = load_emd_features(path, metadata, record_ids=np.array([1, 2]))
    assert record_ids.tolist() == [1]
    assert features.shape[0] == 1
    assert incomplete == [2]
    try:
        load_emd_features(path, metadata, record_ids=np.array([1, 2]), missing_record_policy='error')
    except ValueError as error:
        assert 'Incomplete EMD records' in str(error)
    else:
        assert False
