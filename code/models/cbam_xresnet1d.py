import torch
import torch.nn as nn

from models.xresnet1d import XResNet1d, ResBlock


class ChannelAttention1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=2, keepdim=True))
        maximum = self.mlp(torch.max(x, dim=2, keepdim=True)[0])
        return torch.sigmoid(avg + maximum)


class TemporalAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def scale(self, x):
        return self.excitation(self.pool(x))

    def forward(self, x):
        return x * self.scale(x)


class CBAMResBlock(ResBlock):
    def __init__(self, *args, **kwargs):
        use_cbam = kwargs.pop('use_cbam', True)
        use_se = kwargs.pop('use_se', False)
        cbam_reduction = kwargs.pop('cbam_reduction', 16)
        cbam_kernel_size = kwargs.pop('cbam_kernel_size', 7)
        se_reduction = kwargs.pop('se_reduction', 16)
        if use_cbam and use_se:
            raise ValueError('use_se and use_cbam cannot both be true')
        super().__init__(*args, **kwargs)
        channels = self.convs[-1][0].out_channels
        self.cbam = nn.Sequential(
            ChannelAttention1d(channels, cbam_reduction),
            TemporalAttention1d(cbam_kernel_size)
        ) if use_cbam else None
        self.se = SqueezeExcitation1d(channels, se_reduction) if use_se else None

    def forward(self, x):
        residual = self.convpath(x)
        if self.cbam is not None:
            residual = residual * self.cbam[0](residual)
            residual = residual * self.cbam[1](residual)
        if self.se is not None:
            residual = self.se(residual)
        return self.act(residual + self.idpath(x))


class CBAMXResNet1DLateFusion(nn.Module):
    def __init__(self, expansion=4, layers=(3, 4, 23, 3), num_classes=2,
                 input_channels=12, input_mode='late_fusion', fusion_type='concat',
                  emd_features=None, feature_hidden_dim=256, feature_embedding_dim=128,
                  feature_dropout=0.3, fusion_hidden_dim=256, fusion_dropout=0.4, use_cbam=True,
                  use_se=False, cbam_reduction=16, cbam_kernel_size=7, se_reduction=16,
                  **kwargs):
        super().__init__()
        if use_cbam and use_se:
            raise ValueError('use_se and use_cbam cannot both be true')
        if input_mode not in ('ecg_only', 'feature_only', 'late_fusion'):
            raise ValueError('input_mode must be ecg_only, feature_only, or late_fusion')
        if fusion_type not in ('concat', 'gated'):
            raise ValueError('fusion_type must be concat or gated')
        if input_channels != 12:
            raise ValueError('input_channels must be 12')
        if input_mode != 'ecg_only' and (emd_features is None or emd_features < 1):
            raise ValueError('emd_features must be provided for feature_only and late_fusion modes')
        self.input_mode = input_mode
        self.fusion_type = fusion_type
        self.input_channels = input_channels
        self.emd_features = emd_features

        if input_mode != 'feature_only':
            backbone = XResNet1d(
                 CBAMResBlock, expansion, list(layers), input_channels=input_channels,
                 num_classes=num_classes, use_cbam=use_cbam, use_se=use_se,
                 cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size,
                 se_reduction=se_reduction,
                 **kwargs
            )
            self.ecg_backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.ecg_embedding_size = 2 * (64 * expansion)
        else:
            self.ecg_backbone = None
            self.ecg_embedding_size = None

        if input_mode != 'ecg_only':
            self.feature_encoder = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(12 * emd_features),
                nn.Linear(12 * emd_features, feature_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(feature_dropout),
                nn.Linear(feature_hidden_dim, feature_embedding_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_encoder = None

        if input_mode == 'ecg_only':
            self.output_layer = nn.Linear(self.ecg_embedding_size, num_classes)
        elif input_mode == 'feature_only':
            self.output_layer = nn.Linear(feature_embedding_dim, num_classes)
        else:
            fusion_size = self.ecg_embedding_size + feature_embedding_dim
            if fusion_type == 'gated':
                self.gate = nn.Linear(fusion_size, fusion_size)
            else:
                self.gate = None
            self.fusion = nn.Sequential(
                nn.Linear(fusion_size, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(fusion_dropout)
            )
            self.output_layer = nn.Linear(fusion_hidden_dim, num_classes)

    def _validate_ecg(self, ecg):
        if ecg is None or ecg.dim() != 3:
            raise ValueError('ecg input must have shape [B, 12, T]')
        if ecg.size(1) != 12:
            raise ValueError('ecg input must have 12 channels')

    def _validate_features(self, features, batch_size):
        if features is None:
            raise ValueError('feature input is required for feature_only and late_fusion modes')
        if features.dim() != 3 or features.size(1) != 12 or features.size(2) != self.emd_features:
            raise ValueError('feature input must have shape [B, 12, F] matching emd_features')
        if features.size(0) != batch_size:
            raise ValueError('ecg and feature batch sizes must match')

    def _encode_ecg(self, ecg):
        self._validate_ecg(ecg)
        feature_map = self.ecg_backbone(ecg)
        return torch.cat([
            torch.max(feature_map, dim=2)[0],
            torch.mean(feature_map, dim=2)
        ], dim=1)

    def forward(self, ecg=None, features=None, return_intermediates=False):
        if self.input_mode == 'ecg_only':
            ecg_embedding = self._encode_ecg(ecg)
            logits = self.output_layer(ecg_embedding)
            return (logits, {'ecg_embedding': ecg_embedding}) if return_intermediates else logits
        if self.input_mode == 'feature_only':
            if features is None:
                raise ValueError('feature input is required for feature_only mode')
            self._validate_features(features, features.size(0))
            feature_embedding = self.feature_encoder(features)
            logits = self.output_layer(feature_embedding)
            return (logits, {'feature_embedding': feature_embedding}) if return_intermediates else logits
        ecg_embedding = self._encode_ecg(ecg)
        self._validate_features(features, ecg_embedding.size(0))
        feature_embedding = self.feature_encoder(features)
        fused = torch.cat([ecg_embedding, feature_embedding], dim=1)
        if self.gate is not None:
            fused = fused * torch.sigmoid(self.gate(fused))
        fusion_embedding = self.fusion(fused)
        logits = self.output_layer(fusion_embedding)
        if return_intermediates:
            return logits, {'ecg_embedding': ecg_embedding, 'feature_embedding': feature_embedding,
                            'fused_embedding': fused, 'fusion_embedding': fusion_embedding}
        return logits

    def get_layer_groups(self):
        if self.ecg_backbone is None:
            return (self.feature_encoder, self.output_layer)
        return (self.ecg_backbone, self.output_layer)

    def get_output_layer(self):
        return self.output_layer

    def set_output_layer(self, layer):
        self.output_layer = layer


class EMDBottleneckEncoder(nn.Module):
    """Strict EMD encoder shared by the new bottleneck-only model family."""
    def __init__(self, emd_features=11, hidden_dim=256, embedding_dim=32, dropout=0.3):
        super().__init__()
        if emd_features < 1 or embedding_dim < 1:
            raise ValueError('emd_features and embedding_dim must be positive')
        self.emd_features = emd_features
        self.embedding_dim = embedding_dim
        width = 12 * emd_features
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(width),
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )

    def forward(self, features):
        if features is None or features.dim() != 3 or features.size(1) != 12 or features.size(2) != self.emd_features:
            raise ValueError('EMD features must have shape [B, 12, {}]'.format(self.emd_features))
        if not torch.isfinite(features).all():
            raise ValueError('EMD features must be finite')
        return self.encoder(features)


def _validate_permutation(permutation, batch_size, device):
    if permutation is None:
        raise ValueError('emd_permutation is required when emd_mode is shuffle')
    permutation = torch.as_tensor(permutation, device=device, dtype=torch.long)
    if permutation.dim() != 1 or permutation.numel() != batch_size:
        raise ValueError('emd_permutation must have shape [B]')
    if not torch.equal(torch.sort(permutation).values, torch.arange(batch_size, device=device)):
        raise ValueError('emd_permutation must contain every batch index exactly once')
    return permutation


class CBAMXResNet1DEMDBottleneckGated(nn.Module):
    """New EMD bottleneck late-fusion model; legacy fusion remains untouched."""
    supports_branch_diagnostics = True

    def __init__(self, expansion=4, layers=(3, 4, 23, 3), num_classes=2, input_channels=12,
                 emd_features=11, feature_hidden_dim=256, emd_embedding_dim=32,
                 feature_dropout=0.3, fusion_hidden_dim=256, fusion_dropout=0.4,
                 emd_gate_max=1.0, emd_gate_init_bias=-2.0, use_cbam=True, use_se=False,
                 cbam_reduction=16, cbam_kernel_size=7, se_reduction=16, **kwargs):
        super().__init__()
        if use_cbam and use_se:
            raise ValueError('use_se and use_cbam cannot both be true')
        if input_channels != 12:
            raise ValueError('input_channels must be 12')
        if emd_gate_max <= 0:
            raise ValueError('emd_gate_max must be positive')
        backbone = XResNet1d(
            CBAMResBlock, expansion, list(layers), input_channels=input_channels,
            num_classes=num_classes, use_cbam=use_cbam, use_se=use_se,
            cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size,
            se_reduction=se_reduction, **kwargs
        )
        self.ecg_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.ecg_embedding_size = 2 * (64 * expansion)
        self.emd_encoder = EMDBottleneckEncoder(emd_features, feature_hidden_dim,
                                                 emd_embedding_dim, feature_dropout)
        self.emd_features = emd_features
        self.emd_embedding_dim = emd_embedding_dim
        self.emd_gate_max = float(emd_gate_max)
        self.gate = nn.Linear(self.ecg_embedding_size + emd_embedding_dim, 1)
        nn.init.constant_(self.gate.bias, emd_gate_init_bias)
        self.fusion = nn.Sequential(
            nn.Linear(self.ecg_embedding_size + emd_embedding_dim, fusion_hidden_dim),
            nn.GELU(), nn.Dropout(fusion_dropout)
        )
        self.output_layer = nn.Linear(fusion_hidden_dim, num_classes)

    def _encode_ecg(self, ecg):
        if ecg is None or ecg.dim() != 3 or ecg.size(1) != 12:
            raise ValueError('ecg input must have shape [B, 12, T]')
        if not torch.isfinite(ecg).all():
            raise ValueError('ecg input must be finite')
        feature_map = self.ecg_backbone(ecg)
        return torch.cat([torch.max(feature_map, dim=2)[0], torch.mean(feature_map, dim=2)], dim=1)

    def forward(self, ecg=None, features=None, emd_mode='normal', emd_permutation=None,
                ecg_mode='normal', return_diagnostics=False):
        if emd_mode not in ('normal', 'zero', 'shuffle'):
            raise ValueError('emd_mode must be normal, zero, or shuffle')
        if ecg_mode not in ('normal', 'zero'):
            raise ValueError('ecg_mode must be normal or zero')
        ecg_embedding = self._encode_ecg(ecg)
        if ecg_mode == 'zero':
            ecg_embedding = torch.zeros_like(ecg_embedding)
        if emd_mode == 'shuffle':
            features = features.index_select(0, _validate_permutation(
                emd_permutation, features.size(0), features.device))
        emd_embedding = self.emd_encoder(features)
        if emd_embedding.size(0) != ecg_embedding.size(0):
            raise ValueError('ecg and EMD batch sizes must match')
        if emd_mode == 'zero':
            emd_embedding = torch.zeros_like(emd_embedding)
        raw_gate = self.gate(torch.cat([ecg_embedding, emd_embedding], dim=1))
        emd_gate = self.emd_gate_max * torch.sigmoid(raw_gate)
        gated_emd = emd_embedding * emd_gate
        fusion_embedding = self.fusion(torch.cat([ecg_embedding, gated_emd], dim=1))
        logits = self.output_layer(fusion_embedding)
        if not return_diagnostics:
            return logits
        return logits, {
            'ecg_embedding': ecg_embedding,
            'emd_embedding': emd_embedding,
            'emd_gate': emd_gate,
            'gated_emd_embedding': gated_emd,
            'fusion_embedding': fusion_embedding,
        }


class EMDOnlyBottleneckMLP(nn.Module):
    """EMD-only control using the same bottleneck encoder as gated fusion."""
    supports_branch_diagnostics = True

    def __init__(self, num_classes=2, emd_features=11, feature_hidden_dim=256,
                 emd_embedding_dim=32, feature_dropout=0.3, **kwargs):
        super().__init__()
        self.emd_encoder = EMDBottleneckEncoder(emd_features, feature_hidden_dim,
                                                 emd_embedding_dim, feature_dropout)
        self.emd_features = emd_features
        self.emd_embedding_dim = emd_embedding_dim
        self.output_layer = nn.Linear(emd_embedding_dim, num_classes)

    def forward(self, ecg=None, features=None, emd_mode='normal', emd_permutation=None,
                ecg_mode='normal', return_diagnostics=False):
        del ecg, ecg_mode
        if emd_mode not in ('normal', 'zero', 'shuffle'):
            raise ValueError('emd_mode must be normal, zero, or shuffle')
        if emd_mode == 'shuffle':
            features = features.index_select(0, _validate_permutation(
                emd_permutation, features.size(0), features.device))
        emd_embedding = self.emd_encoder(features)
        if emd_mode == 'zero':
            emd_embedding = torch.zeros_like(emd_embedding)
        logits = self.output_layer(emd_embedding)
        if not return_diagnostics:
            return logits
        return logits, {'emd_embedding': emd_embedding}


def cbam_xresnet1d101(**kwargs):
    expansion = kwargs.pop('expansion', 4)
    layers = kwargs.pop('layers', (3, 4, 23, 3))
    return CBAMXResNet1DLateFusion(expansion=expansion, layers=layers, **kwargs)


def build_model(backbone_name, num_classes, use_cbam=False, use_se=False, use_emd=False,
                  fusion_type=None, emd_features=11, feature_hidden_dim=256,
                  feature_embedding_dim=128, fusion_hidden_dim=256, model_variant=None,
                  emd_embedding_dim=None, **kwargs):
    if backbone_name != 'xresnet1d101':
        raise ValueError('Unsupported backbone: {}'.format(backbone_name))
    if use_cbam and use_se:
        raise ValueError('use_se and use_cbam cannot both be true')
    if model_variant == 'emd_bottleneck_gated':
        return CBAMXResNet1DEMDBottleneckGated(
            num_classes=num_classes, use_cbam=use_cbam, use_se=use_se,
            emd_features=emd_features, feature_hidden_dim=feature_hidden_dim,
            emd_embedding_dim=emd_embedding_dim or feature_embedding_dim,
            fusion_hidden_dim=fusion_hidden_dim, **kwargs)
    if model_variant == 'emd_only_bottleneck':
        return EMDOnlyBottleneckMLP(
            num_classes=num_classes, emd_features=emd_features,
            feature_hidden_dim=feature_hidden_dim,
            emd_embedding_dim=emd_embedding_dim or feature_embedding_dim, **kwargs)
    if use_emd:
        input_mode = 'late_fusion'
        fusion_type = fusion_type or 'concat'
    else:
        input_mode = 'ecg_only'
        fusion_type = 'concat'
    return cbam_xresnet1d101(
        num_classes=num_classes, input_mode=input_mode, use_cbam=use_cbam, use_se=use_se,
        fusion_type=fusion_type, emd_features=emd_features,
        feature_hidden_dim=feature_hidden_dim,
        feature_embedding_dim=feature_embedding_dim,
        fusion_hidden_dim=fusion_hidden_dim, **kwargs
    )


def cbam_xresnet1d101_emd_late_fusion_emb32_gated(**kwargs):
    kwargs.setdefault('model_variant', 'emd_bottleneck_gated')
    kwargs.setdefault('emd_embedding_dim', 32)
    kwargs.setdefault('use_cbam', True)
    return build_model('xresnet1d101', **kwargs)


def cbam_xresnet1d101_emd_late_fusion_emb64_gated(**kwargs):
    kwargs.setdefault('model_variant', 'emd_bottleneck_gated')
    kwargs.setdefault('emd_embedding_dim', 64)
    kwargs.setdefault('use_cbam', True)
    return build_model('xresnet1d101', **kwargs)


def emd_only_mlp_emb32(**kwargs):
    kwargs.setdefault('model_variant', 'emd_only_bottleneck')
    kwargs.setdefault('emd_embedding_dim', 32)
    return build_model('xresnet1d101', **kwargs)


def emd_only_mlp_emb64(**kwargs):
    kwargs.setdefault('model_variant', 'emd_only_bottleneck')
    kwargs.setdefault('emd_embedding_dim', 64)
    return build_model('xresnet1d101', **kwargs)
