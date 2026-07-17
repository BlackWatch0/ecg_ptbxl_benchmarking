import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


MARKERS = {
    'clean': 'ptbxl_database_clean_no_noise.csv',
    'noisy': 'ptbxl_noisy_mixed_shared_manifest.csv',
    'emd': 'PTBXL_Batch_Original_EMD_reduced_features.csv',
}
RAW_CLEAN_MARKER = 'ptbxl_database.csv'
RAW_NOISY_MARKER = 'ptbxl_original_database_plus_mixed_manifest.csv'
TARGETS = {
    'clean': 'ptbxl_clean_no_noise',
    'noisy': 'ptbxl_noisy_mixed_shared',
    'emd': 'emd_features',
}


def extract(archive, destination, asset):
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as file:
            file.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as file:
            file.extractall(destination)
    else:
        try:
            import py7zr
            is_7z = py7zr.is_7zfile(archive)
        except ImportError:
            is_7z = False
        if is_7z:
            with py7zr.SevenZipFile(archive, mode='r') as file:
                file.extractall(destination)
            return
        try:
            import rarfile
            is_rar = rarfile.is_rarfile(archive)
        except ImportError:
            is_rar = False
        if is_rar:
            try:
                with rarfile.RarFile(archive) as file:
                    file.extractall(destination)
            except rarfile.RarCannotExec:
                subprocess.check_call(['7z', 'x', '-y', '-o{}'.format(destination), str(archive)])
        elif asset == 'emd':
            try:
                header = Path(archive).open('r', encoding='utf-8').readline()
            except UnicodeDecodeError:
                header = ''
            if 'RecordNumber' not in header:
                raise ValueError('Unsupported archive format: {}'.format(archive))
            target = destination / 'emd_features' / 'original'
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, target / MARKERS['emd'])
        else:
            raise ValueError('Unsupported archive format: {}'.format(archive))


def locate(root, marker, asset):
    matches = list(Path(root).rglob(marker))
    if asset == 'clean' and not matches:
        # Active PTB-XL releases use the upstream metadata filename. The EMD
        # workflow expects an older compatibility name but uses the same rows.
        matches = list(Path(root).rglob(RAW_CLEAN_MARKER))
    if asset == 'noisy' and not matches:
        matches = list(Path(root).rglob(RAW_NOISY_MARKER))
    if len(matches) != 1:
        raise ValueError('Expected one {} marker {}, found {}'.format(asset, marker, matches))
    if asset == 'emd':
        return matches[0].parent.parent
    return matches[0].parent


def prepare(asset, archive, data_root, workspace, replace=False):
    data_root = Path(data_root)
    target = data_root / TARGETS[asset]
    if target.exists() and not replace:
        print('{} already exists: {}'.format(asset, target))
        return
    staging = Path(workspace) / '{}_extracted'.format(asset)
    if staging.exists():
        shutil.rmtree(staging)
    extract(archive, staging, asset)
    source = locate(staging, MARKERS[asset], asset)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    if asset == 'clean' and not (target / MARKERS['clean']).exists():
        raw_metadata = target / RAW_CLEAN_MARKER
        if not raw_metadata.exists():
            raise FileNotFoundError('Active clean PTB-XL metadata is missing: {}'.format(raw_metadata))
        shutil.copy2(raw_metadata, target / MARKERS['clean'])
        print('Created EMD compatibility metadata from active PTB-XL: {}'.format(target / MARKERS['clean']))
    if asset == 'noisy' and not (target / MARKERS['noisy']).exists():
        raw_manifest = target / RAW_NOISY_MARKER
        if not raw_manifest.exists():
            raise FileNotFoundError('Active noisy PTB-XL manifest is missing: {}'.format(raw_manifest))
        shutil.copy2(raw_manifest, target / MARKERS['noisy'])
        print('Created EMD compatibility manifest from active noisy PTB-XL: {}'.format(target / MARKERS['noisy']))
    print('Prepared {}: {}'.format(asset, target))


def validate(data_root):
    data_root = Path(data_root)
    expected = {
        'clean': data_root / TARGETS['clean'] / MARKERS['clean'],
        'noisy': data_root / TARGETS['noisy'] / MARKERS['noisy'],
        'emd': data_root / TARGETS['emd'] / 'original' / MARKERS['emd'],
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing prepared datasets: {}. Expected: {}'.format(missing, expected))
    print('Dataset validation passed')
    for name, path in expected.items():
        print('{}: {}'.format(name, path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'validate'])
    parser.add_argument('--asset', choices=sorted(MARKERS))
    parser.add_argument('--archive')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--workspace', default='/content/ecg_downloads')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.data_root)
    else:
        if not args.asset or not args.archive:
            raise ValueError('prepare requires --asset and --archive')
        prepare(args.asset, args.archive, args.data_root, args.workspace, args.replace)


if __name__ == '__main__':
    main()
