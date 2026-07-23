import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.cbam_xresnet1d import (CBAMXResNet1DEMDBottleneckGated,
                                   CBAMXResNet1DLateFusion, EMDOnlyBottleneckMLP,
                                   build_model)


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


def test_emd_bottleneck_gated_modes_diagnostics_and_backward():
    ecg = torch.randn(4, 12, 250)
    features = torch.randn(4, 12, 11)
    model = CBAMXResNet1DEMDBottleneckGated(
        expansion=1, layers=(1, 1, 1, 1), num_classes=5, emd_features=11,
        feature_hidden_dim=16, emd_embedding_dim=32, fusion_hidden_dim=16,
        emd_gate_max=.75).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    logits, diagnostics = model(ecg, features, return_diagnostics=True)
    shuffled, shuffled_diagnostics = model(
        ecg, features, emd_mode='shuffle', emd_permutation=permutation, return_diagnostics=True)
    repeated = model(ecg, features, emd_mode='shuffle', emd_permutation=permutation)
    zero_emd = model(ecg, features, emd_mode='zero')
    zero_ecg = model(ecg, features, ecg_mode='zero')
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()
    assert torch.equal(shuffled, repeated)
    assert shuffled_diagnostics['emd_embedding'].shape == (4, 32)
    assert diagnostics['emd_gate'].shape == (4, 1)
    assert torch.all(diagnostics['emd_gate'] >= 0)
    assert torch.all(diagnostics['emd_gate'] <= .75)
    assert not torch.equal(logits, zero_emd)
    assert not torch.equal(logits, zero_ecg)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert any(parameter.grad is not None for parameter in model.emd_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.gate.parameters())


def test_emd_bottleneck_feature_only_and_factory_variants():
    features = torch.randn(3, 12, 11)
    control = EMDOnlyBottleneckMLP(num_classes=5, emd_features=11, feature_hidden_dim=16,
                                   emd_embedding_dim=64)
    logits, diagnostics = control(features=features, return_diagnostics=True)
    assert logits.shape == (3, 5)
    assert diagnostics['emd_embedding'].shape == (3, 64)
    model = build_model('xresnet1d101', 5, use_cbam=True, use_emd=True,
                        model_variant='emd_bottleneck_gated', emd_embedding_dim=32,
                        emd_features=11, feature_hidden_dim=16, fusion_hidden_dim=16,
                        expansion=1, layers=(1, 1, 1, 1))
    assert model.emd_embedding_dim == 32
    legacy = _model('late_fusion', 'concat', True)
    restored = _model('late_fusion', 'concat', True)
    restored.load_state_dict(legacy.state_dict(), strict=True)
