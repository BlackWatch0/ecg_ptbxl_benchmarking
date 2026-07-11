import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.cbam_xresnet1d import CBAMXResNet1DLateFusion


def _model(input_mode, fusion_type='concat', use_cbam=True):
    return CBAMXResNet1DLateFusion(
        expansion=1, layers=(1, 1, 1, 1), num_classes=5,
        input_mode=input_mode, fusion_type=fusion_type, emd_features=11,
        use_cbam=use_cbam, feature_hidden_dim=16, feature_embedding_dim=8,
        fusion_hidden_dim=16
    )


def test_cbam_xresnet1d_forward_modes_and_backward():
    ecg = torch.randn(4, 12, 250)
    features = torch.randn(4, 12, 11)
    for input_mode, fusion_type, use_cbam in [
        ('ecg_only', 'concat', True),
        ('feature_only', 'concat', True),
        ('late_fusion', 'concat', True),
        ('late_fusion', 'gated', True),
        ('late_fusion', 'concat', False),
    ]:
        model = _model(input_mode, fusion_type, use_cbam)
        logits = model(ecg, features)
        assert logits.shape == (4, 5)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        if input_mode == 'late_fusion':
            assert any(parameter.grad is not None for parameter in model.feature_encoder.parameters())
        if use_cbam and input_mode != 'feature_only':
            assert any('cbam' in name and parameter.grad is not None for name, parameter in model.named_parameters())
