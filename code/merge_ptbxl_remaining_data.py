import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd


def extract_archive(archive, staging):
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive) as file:
        root = staging.resolve()
        for member in file.getmembers():
            path = (staging / member.name).resolve()
            if root not in path.parents and path != root:
                raise ValueError('Archive contains unsafe path: {}'.format(member.name))
        file.extractall(staging)


def locate_content_root(staging, kind):
    marker = 'ptbxl_database' if kind == 'clean' else 'manifest'
    matches = [path for path in staging.rglob('*.csv') if marker in path.name]
    if len(matches) == 1:
        return matches[0].parent
    records = list(staging.rglob('records100'))
    if len(records) == 1:
        return records[0].parent
    children = [path for path in staging.iterdir() if not path.name.startswith('.')]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    raise ValueError('Cannot identify {} patch root in {}'.format(kind, staging))


def merge_csv(existing, patch, key_columns):
    before = pd.read_csv(existing) if existing.exists() else pd.DataFrame()
    incoming = pd.read_csv(patch)
    missing = [column for column in key_columns if column not in incoming.columns]
    if missing:
        raise ValueError('{} is missing merge key columns {}'.format(patch, missing))
    if len(before):
        missing = [column for column in key_columns if column not in before.columns]
        if missing:
            raise ValueError('{} is missing merge key columns {}'.format(existing, missing))
    merged = pd.concat([before, incoming], ignore_index=True)
    duplicates = merged.duplicated(key_columns, keep=False)
    duplicate_count = int(duplicates.sum())
    merged = merged.drop_duplicates(key_columns, keep='first').sort_values(key_columns).reset_index(drop=True)
    merged.to_csv(existing, index=False)
    return dict(existing_rows=len(before), patch_rows=len(incoming), merged_rows=len(merged),
                duplicate_rows=duplicate_count)


def merge_records_file(existing, patch):
    old_lines = existing.read_text().splitlines() if existing.exists() else []
    new_lines = patch.read_text().splitlines()
    merged = sorted(set(old_lines).union(new_lines))
    existing.write_text('\n'.join(merged) + '\n')
    return dict(existing_lines=len(old_lines), patch_lines=len(new_lines), merged_lines=len(merged))


def destination_csv(target, source, kind):
    if kind == 'clean' and source.name.startswith('ptbxl_database'):
        return target / 'ptbxl_database_clean_no_noise.csv', ['ecg_id']
    if kind == 'noisy' and 'manifest' in source.name:
        return target / 'ptbxl_noisy_mixed_shared_manifest.csv', ['ecg_id', 'snr_target_db']
    return None, None


def merge_patch(archive, target, kind, workspace):
    archive, target = Path(archive), Path(target)
    if not archive.exists() or not target.exists():
        raise FileNotFoundError('Missing archive or target: {} {}'.format(archive, target))
    staging = Path(workspace) / '{}_remaining_patch'.format(kind)
    extract_archive(archive, staging)
    source_root = locate_content_root(staging, kind)
    report = dict(archive=str(archive), source_root=str(source_root), copied_files=0, merged_csv={},
                  merged_records={})
    for source in source_root.rglob('*'):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if source.name in ('raw100.npy', 'raw500.npy'):
            continue
        destination, keys = destination_csv(target, source, kind)
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            report['merged_csv'][source.name] = merge_csv(destination, source, keys)
            continue
        if source.name == 'RECORDS':
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            report['merged_records'][str(relative)] = merge_records_file(destination, source)
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size != source.stat().st_size:
            raise ValueError('Refusing to overwrite conflicting file: {}'.format(destination))
        if not destination.exists():
            shutil.copy2(source, destination)
            report['copied_files'] += 1
    if kind == 'clean':
        for name in ('raw100.npy', 'raw500.npy'):
            cache = target / name
            if cache.exists():
                cache.unlink()
                report['invalidated_cache'] = report.get('invalidated_cache', []) + [str(cache)]
    return report


def validate(data_root):
    data_root = Path(data_root)
    clean_root = data_root / 'ptbxl_clean_no_noise'
    noisy_root = data_root / 'ptbxl_noisy_mixed_shared'
    emd_path = data_root / 'emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv'
    metadata = pd.read_csv(clean_root / 'ptbxl_database_clean_no_noise.csv')
    if metadata.ecg_id.duplicated().any():
        raise ValueError('Clean metadata contains duplicate ecg_id values')
    missing_waveforms = []
    for row in metadata.itertuples():
        if not (clean_root / (row.filename_lr + '.hea')).exists():
            missing_waveforms.append(int(row.ecg_id))
    if missing_waveforms:
        raise ValueError('Missing clean WFDB records ({}): {}'.format(len(missing_waveforms), missing_waveforms[:20]))
    manifest = pd.read_csv(noisy_root / 'ptbxl_noisy_mixed_shared_manifest.csv')
    key = ['ecg_id', 'snr_target_db']
    if manifest.duplicated(key).any():
        raise ValueError('Noisy manifest contains duplicate ecg_id/SNR rows')
    clean_ids = set(metadata.ecg_id)
    snr_counts = {}
    for snr in (24, 12, 6, 0, -6):
        ids = set(manifest.loc[manifest.snr_target_db == snr, 'ecg_id'])
        missing = clean_ids.difference(ids)
        if missing:
            raise ValueError('Noisy SNR {} is missing {} clean IDs: {}'.format(snr, len(missing), sorted(missing)[:20]))
        snr_counts[str(snr)] = len(ids)
    emd_ids = set(pd.read_csv(emd_path, usecols=['RecordNumber']).RecordNumber.unique())
    missing_emd = clean_ids.difference(emd_ids)
    if missing_emd:
        raise ValueError('Original EMD is missing {} clean IDs: {}'.format(len(missing_emd), sorted(missing_emd)[:20]))
    return dict(clean_metadata_records=len(metadata), clean_waveform_records=len(metadata) - len(missing_waveforms),
                noisy_records_per_snr=snr_counts, original_emd_record_coverage=len(clean_ids.intersection(emd_ids)),
                missing_clean_waveforms=missing_waveforms, missing_original_emd_ids=sorted(missing_emd))


def main():
    parser = argparse.ArgumentParser(description='Merge missing original/noisy PTB-XL records into prepared datasets.')
    parser.add_argument('action', choices=['merge', 'validate'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--clean-archive')
    parser.add_argument('--noisy-archive')
    parser.add_argument('--workspace', default='/content/ecg_downloads')
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if args.action == 'merge':
        if not args.clean_archive or not args.noisy_archive:
            raise ValueError('merge requires --clean-archive and --noisy-archive')
        result = {
            'clean': merge_patch(args.clean_archive, data_root / 'ptbxl_clean_no_noise', 'clean', args.workspace),
            'noisy': merge_patch(args.noisy_archive, data_root / 'ptbxl_noisy_mixed_shared', 'noisy', args.workspace),
        }
        result['validation'] = validate(data_root)
        path = data_root / 'full_data_merge_report.json'
        with open(path, 'w') as file:
            json.dump(result, file, indent=2)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(validate(data_root), indent=2))


if __name__ == '__main__':
    main()
