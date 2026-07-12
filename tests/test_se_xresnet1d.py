import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from models.cbam_xresnet1d import CBAMResBlock, SqueezeExcitation1d, build_model


def _small_model(use_se=True, use_emd=False):
    return build_model(
        'xresnet1d101', 5, use_se=use_se, use_emd=use_emd,
        emd_features=11, expansion=1, layers=(1, 1, 1, 1),
        feature_hidden_dim=16, feature_embedding_dim=8, fusion_hidden_dim=16
    )


def test_se_scale_and_residual_shapes():
    x = torch.randn(3, 8, 31)
    se = SqueezeExcitation1d(8)
    assert se.scale(x).shape == (3, 8, 1)
    assert se(x).shape == x.shape

    block = CBAMResBlock(1, 8, 8, ndim=1, use_cbam=False, use_se=True)
    assert block(x).shape == x.shape


def test_se_ecg_only_forward_bce_and_backward_without_emd():
    model = _small_model()
    logits = model(torch.randn(2, 12, 250))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert any('se.' in name and parameter.grad is not None
               for name, parameter in model.named_parameters())


def test_se_late_fusion_uses_emd():
    model = _small_model(use_emd=True).eval()
    ecg = torch.randn(2, 12, 250)
    features = torch.randn(2, 12, 11)
    first = model(ecg, features)
    second = model(ecg, -features)
    assert not torch.allclose(first, second)


def test_se_factory_validation_and_parameter_count():
    baseline = _small_model(use_se=False)
    se_model = _small_model()
    assert sum(p.numel() for p in se_model.parameters()) > sum(p.numel() for p in baseline.parameters())
    with pytest.raises(ValueError, match='cannot both be true'):
        build_model('xresnet1d101', 5, use_se=True, use_cbam=True)
