import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


MARKERS = {
    'clean': 'ptbxl_database_clean_no_noise.csv',
    'noisy': 'ptbxl_noisy_mixed_shared_manifest.csv',
    'emd': 'PTBXL_Batch_Original_EMD_reduced_features.csv',
}
TARGETS = {
    'clean': 'ptbxl_clean_no_noise',
    'noisy': 'ptbxl_noisy_mixed_shared',
    'emd': 'emd_features',
}


def extract(archive, destination):
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
        raise ValueError('Unsupported archive format: {}'.format(archive))


def locate(root, marker, asset):
    matches = list(Path(root).rglob(marker))
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
    extract(archive, staging)
    source = locate(staging, MARKERS[asset], asset)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
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
