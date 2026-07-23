import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.cbam_xresnet1d import build_model
from run_ablation_study import ECGDataset, train_model


def test_ablation_model_factory_modes():
    ecg = torch.randn(2, 12, 1000)
    emd = torch.randn(2, 12, 11)
    for use_cbam, use_se, use_emd in [
        (False, False, False), (True, False, False), (False, False, True),
        (True, False, True), (False, True, False), (False, True, True)
    ]:
        model = build_model('xresnet1d101', 5, use_cbam=use_cbam, use_se=use_se, use_emd=use_emd,
                             fusion_type='concat', emd_features=11)
        logits = model(ecg, emd if use_emd else None)
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()


def test_ablation_history_records_train_and_validation_accuracy(tmp_path):
    class TinyFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(13, 5)

        def forward(self, ecg, features):
            return self.output(torch.cat([ecg.mean(dim=2), features.mean(dim=2)[:, :1]], dim=1))

    ecg = np.zeros((4, 32, 12), dtype=np.float32)
    emd = np.zeros((4, 12, 11), dtype=np.float32)
    labels = np.eye(5, dtype=np.float32)[:4]
    loader = DataLoader(ECGDataset(ecg, labels, emd), batch_size=2)
    checkpoint = tmp_path / 'checkpoint.pth'
    history = tmp_path / 'history.csv'
    config = {'learning_rate': .01, 'weight_decay': .01, 'epochs': 1,
              'mixed_precision': False, 'use_emd': True, 'experiment_name': 'tiny'}
    train_model(TinyFusion(), loader, loader, config, torch.device('cpu'), checkpoint, history)
    frame = pd.read_csv(history)
    assert {'train_accuracy', 'valid_accuracy'}.issubset(frame.columns)
    assert frame.train_accuracy.between(0, 1).all()
    assert frame.valid_accuracy.between(0, 1).all()
