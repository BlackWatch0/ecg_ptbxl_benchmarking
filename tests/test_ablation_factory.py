import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.cbam_xresnet1d import build_model


def test_ablation_model_factory_modes():
    ecg = torch.randn(2, 12, 1000)
    emd = torch.randn(2, 12, 11)
    for use_cbam, use_emd in [(False, False), (True, False), (False, True), (True, True)]:
        model = build_model('xresnet1d101', 5, use_cbam=use_cbam, use_emd=use_emd,
                            fusion_type='concat', emd_features=11)
        logits = model(ecg, emd if use_emd else None)
        assert logits.shape == (2, 5)
        assert torch.isfinite(logits).all()
