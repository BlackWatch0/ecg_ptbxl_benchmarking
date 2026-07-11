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


class CBAMResBlock(ResBlock):
    def __init__(self, *args, **kwargs):
        use_cbam = kwargs.pop('use_cbam', True)
        cbam_reduction = kwargs.pop('cbam_reduction', 16)
        cbam_kernel_size = kwargs.pop('cbam_kernel_size', 7)
        super().__init__(*args, **kwargs)
        channels = self.convs[-1][0].out_channels
        self.cbam = nn.Sequential(
            ChannelAttention1d(channels, cbam_reduction),
            TemporalAttention1d(cbam_kernel_size)
        ) if use_cbam else None

    def forward(self, x):
        residual = self.convpath(x)
        if self.cbam is not None:
            residual = residual * self.cbam[0](residual)
            residual = residual * self.cbam[1](residual)
        return self.act(residual + self.idpath(x))


class CBAMXResNet1DLateFusion(nn.Module):
    def __init__(self, expansion=4, layers=(3, 4, 23, 3), num_classes=2,
                 input_channels=12, input_mode='late_fusion', fusion_type='concat',
                 emd_features=None, feature_hidden_dim=256, feature_embedding_dim=128,
                 feature_dropout=0.3, fusion_hidden_dim=256, fusion_dropout=0.4, use_cbam=True,
                 cbam_reduction=16, cbam_kernel_size=7, **kwargs):
        super().__init__()
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
                num_classes=num_classes, use_cbam=use_cbam,
                cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size,
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

    def forward(self, ecg=None, features=None):
        if self.input_mode == 'ecg_only':
            return self.output_layer(self._encode_ecg(ecg))
        if self.input_mode == 'feature_only':
            if features is None:
                raise ValueError('feature input is required for feature_only mode')
            self._validate_features(features, features.size(0))
            return self.output_layer(self.feature_encoder(features))
        ecg_embedding = self._encode_ecg(ecg)
        self._validate_features(features, ecg_embedding.size(0))
        feature_embedding = self.feature_encoder(features)
        fused = torch.cat([ecg_embedding, feature_embedding], dim=1)
        if self.gate is not None:
            fused = fused * torch.sigmoid(self.gate(fused))
        return self.output_layer(self.fusion(fused))

    def get_layer_groups(self):
        if self.ecg_backbone is None:
            return (self.feature_encoder, self.output_layer)
        return (self.ecg_backbone, self.output_layer)

    def get_output_layer(self):
        return self.output_layer

    def set_output_layer(self, layer):
        self.output_layer = layer


def cbam_xresnet1d101(**kwargs):
    return CBAMXResNet1DLateFusion(expansion=4, layers=(3, 4, 23, 3), **kwargs)


def build_model(backbone_name, num_classes, use_cbam=False, use_emd=False,
                fusion_type=None, emd_features=11, feature_hidden_dim=256,
                feature_embedding_dim=128, fusion_hidden_dim=256, **kwargs):
    if backbone_name != 'xresnet1d101':
        raise ValueError('Unsupported backbone: {}'.format(backbone_name))
    if use_emd:
        input_mode = 'late_fusion'
        fusion_type = fusion_type or 'concat'
    else:
        input_mode = 'ecg_only'
        fusion_type = 'concat'
    return cbam_xresnet1d101(
        num_classes=num_classes, input_mode=input_mode, use_cbam=use_cbam,
        fusion_type=fusion_type, emd_features=emd_features,
        feature_hidden_dim=feature_hidden_dim,
        feature_embedding_dim=feature_embedding_dim,
        fusion_hidden_dim=fusion_hidden_dim, **kwargs
    )
