import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
FEATURES = ('cA4_Mean', 'cA4_Std', 'cA4_Energy', 'cD2_cD3_Mean', 'cD2_cD3_Std', 'cD2_cD3_Energy')
FEATURE_COLUMNS = tuple('Lead_{}_{}'.format(lead, feature) for lead in LEADS for feature in FEATURES)
RECORD_ID_PATTERN = re.compile(r'(?P<record_id>\d{5}_lr)')
CACHE_VERSION = 1


def canonicalize_wavelet_record_id(raw_id):
    value = Path(str(raw_id).replace('\\', '/')).name
    value = re.sub(r'\.(mat|hea|dat|csv)$', '', value, flags=re.IGNORECASE)
    matches = RECORD_ID_PATTERN.findall(value)
    if len(matches) != 1:
        if len(matches) > 1:
            raise ValueError('Ambiguous Wavelet record ID {!r}: {}'.format(raw_id, matches))
        raise ValueError('Cannot parse Wavelet record ID {!r}'.format(raw_id))
    return matches[0]


def _schema(columns):
    duplicate = sorted({name for name in columns if columns.count(name) > 1})
    identifier = next((name for name in ('RecordName', 'FileName') if name in columns), None)
    features = [name for name in columns if name != identifier]
    missing = [name for name in FEATURE_COLUMNS if name not in features]
    extra = [name for name in features if name not in FEATURE_COLUMNS]
    return {
        'identifier': identifier, 'column_count': len(columns), 'feature_count': len(features),
        'expected_columns': list(FEATURE_COLUMNS), 'actual_feature_columns': features,
        'missing_columns': missing, 'extra_columns': extra, 'duplicate_columns': duplicate,
        'column_order_matches': features == list(FEATURE_COLUMNS),
        'valid': identifier is not None and len(columns) == 73 and not missing and not extra and not duplicate and features == list(FEATURE_COLUMNS),
    }


class WaveletFeatureStore:
    def __init__(self, root_dir, schema_file=None, cache_dir=None, strict=True):
        self.root_dir = Path(root_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.strict = strict
        if not self.root_dir.is_dir():
            raise FileNotFoundError('Wavelet root does not exist: {}'.format(self.root_dir))

    def _files(self, archive_section, snr):
        sections = [path for path in self.root_dir.rglob(archive_section) if path.is_dir()]
        if len(sections) != 1:
            raise ValueError('Expected one Wavelet section {}, found {}'.format(archive_section, sections))
        section = sections[0]
        if snr is None:
            files = sorted(section.rglob('*.csv'))
        else:
            expected = 'SNRM{}dB'.format(abs(snr)) if snr < 0 else 'SNR{}dB'.format(snr)
            folders = [path for path in section.rglob(expected) if path.is_dir()]
            if len(folders) != 1:
                raise ValueError('Expected one Wavelet SNR folder {} in {}'.format(expected, section))
            files = sorted(folders[0].glob('*.csv'))
        if len(files) != 22:
            raise ValueError('{} SNR {} requires exactly 22 CSV files; found {}'.format(archive_section, snr, len(files)))
        return files

    def load_scenario(self, archive_section, snr=None):
        files = self._files(archive_section, snr)
        frames = []
        source_state = []
        for path in files:
            frame = pd.read_csv(path)
            check = _schema(frame.columns.tolist())
            if not check['valid']:
                raise ValueError('Invalid Wavelet schema in {}: {}'.format(path, check))
            ids = [canonicalize_wavelet_record_id(value) for value in frame[check['identifier']]]
            values = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('NaN or Inf Wavelet values in {}'.format(path))
            frames.append(pd.DataFrame(values, columns=FEATURE_COLUMNS).assign(record_id=ids))
            source_state.append({'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
        combined = pd.concat(frames, ignore_index=True)
        duplicate = sorted(combined.loc[combined.record_id.duplicated(False), 'record_id'].unique())
        if duplicate:
            raise ValueError('Duplicate Wavelet IDs in {} SNR {}: {}'.format(archive_section, snr, duplicate[:10]))
        values = combined.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        report = {
            'archive_section': archive_section, 'snr': snr, 'raw_row_count': len(combined),
            'unique_id_count': int(combined.record_id.nunique()), 'duplicate_ids': duplicate,
            'parse_failure_count': 0, 'ambiguous_id_count': 0, 'feature_columns': list(FEATURE_COLUMNS),
            'schema_hash': hashlib.sha256('\n'.join(FEATURE_COLUMNS).encode()).hexdigest(),
            'sources': source_state, 'nan_count': int(np.isnan(values).sum()), 'inf_count': int(np.isinf(values).sum()),
            'all_zero_columns': [FEATURE_COLUMNS[i] for i in np.flatnonzero(np.all(values == 0, axis=0))],
            'constant_columns': [FEATURE_COLUMNS[i] for i in np.flatnonzero(np.ptp(values, axis=0) == 0)],
        }
        return combined.record_id.to_numpy(), values.reshape(-1, 12, 6), report

    def get_features(self, record_ids, archive_section, snr=None):
        ids, features, report = self.load_scenario(archive_section, snr)
        mapping = {record_id: position for position, record_id in enumerate(ids)}
        missing = [record_id for record_id in record_ids if record_id not in mapping]
        if missing:
            raise ValueError('Missing Wavelet IDs: {}'.format(missing[:20]))
        return np.asarray([features[mapping[record_id]] for record_id in record_ids], dtype=np.float32), report

    def cache(self, name, record_ids, features, report):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (name + '.npz')
        np.savez_compressed(path, record_ids=np.asarray(record_ids), features=np.asarray(features, dtype=np.float32),
                            feature_columns=np.asarray(FEATURE_COLUMNS))
        manifest = self.cache_dir / 'cache_manifest.json'
        entries = json.loads(manifest.read_text()) if manifest.exists() else {}
        entries[path.name] = dict(report, cache_version=CACHE_VERSION)
        manifest.write_text(json.dumps(entries, indent=2))
