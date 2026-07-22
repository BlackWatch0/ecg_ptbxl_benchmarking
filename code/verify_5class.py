import ast
from pathlib import Path

import numpy as np
import pandas as pd

from utils import utils
from utils import data_assets

DATA_ROOT = Path('../data')
CLEAN_ROOT = DATA_ROOT / 'ptbxl_clean_no_noise'


def main():
    clean_root = data_assets.clean_dataset_root(DATA_ROOT)
    metadata = data_assets.load_metadata(clean_root, 'ptbxl_database_clean_no_noise.csv')
    labels = utils.compute_label_aggregations(metadata.copy(), str(clean_root) + '/', 'superdiagnostic')
    _, labels, y, mlb = utils.select_data(
        np.empty(len(labels), dtype=object), labels, 'superdiagnostic', 0, '/tmp/',
        class_order=['NORM', 'MI', 'STTC', 'CD', 'HYP']
    )
    assert y.shape[1] == 5, 'Label shape must be (N,5)'
    assert mlb.classes_.tolist() == ['NORM', 'MI', 'STTC', 'CD', 'HYP'], 'Wrong class order'
    assert set(np.unique(y)).issubset({0, 1}), 'Labels must be 0/1'
    tr = y[labels.strat_fold <= 8]
    va = y[labels.strat_fold == 9]
    te = y[labels.strat_fold == 10]
    print('verified_5class task=superdiagnostic classes={}'.format(mlb.classes_.tolist()))
    print('train/val/test: {} / {} / {}'.format(len(tr), len(va), len(te)))
    print('class counts:', dict(zip(mlb.classes_, y.sum(axis=0).astype(int).tolist())))
    print('multi_label_pct: {:.1f}%'.format((y.sum(axis=1) > 1).mean() * 100))
    print('positive_rate: {:.4f}'.format(float(y.mean())))
    print('mean_positive_labels: {:.4f}'.format(float(y.sum(axis=1).mean())))


if __name__ == '__main__':
    main()
