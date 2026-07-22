import torch
import torch.nn as nn

from models.cbam_xresnet1d import CBAMResBlock
from models.xresnet1d import ResBlock, XResNet1d


class WaveletFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim=256, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, features):
        if features.dim() != 3 or tuple(features.shape[1:]) != (12, 6):
            raise ValueError('Wavelet features must have shape [B, 12, 6]')
        return self.network(features)


class WaveletLateFusionXResNet(nn.Module):
    def __init__(self, num_classes=5, use_cbam=False, input_channels=12,
                 feature_hidden_dim=256, feature_embedding_dim=128,
                 feature_dropout=0.2, fusion_hidden_dim=256,
                 fusion_dropout=0.2):
        super().__init__()
        if input_channels != 12:
            raise ValueError('PTB-XL Wavelet ablation requires 12 ECG channels')
        block = CBAMResBlock if use_cbam else ResBlock
        block_args = dict(use_cbam=True) if use_cbam else {}
        backbone = XResNet1d(
            block, 4, [3, 4, 23, 3], input_channels=input_channels,
            num_classes=num_classes, kernel_size=5, ps_head=0.5,
            lin_ftrs_head=[128], **block_args
        )
        self.ecg_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.ecg_embedding_dim = 512
        self.use_wavelet = True
        self.use_cbam = use_cbam
        self.wavelet_encoder = WaveletFeatureEncoder(
            feature_hidden_dim, feature_embedding_dim, feature_dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.ecg_embedding_dim + feature_embedding_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def encode_ecg(self, ecg):
        if ecg.dim() != 3 or ecg.shape[1] != 12 or ecg.shape[2] != 1000:
            raise ValueError('ECG input must have shape [B, 12, 1000]')
        feature_map = self.ecg_backbone(ecg)
        return torch.cat((feature_map.max(dim=2).values, feature_map.mean(dim=2)), dim=1)

    def forward(self, ecg, features):
        ecg_embedding = self.encode_ecg(ecg)
        wavelet_embedding = self.wavelet_encoder(features)
        if ecg_embedding.shape[0] != wavelet_embedding.shape[0]:
            raise ValueError('ECG and Wavelet batch sizes must match')
        return self.fusion(torch.cat((ecg_embedding, wavelet_embedding), dim=1))


class ECGOnlyXResNet(nn.Module):
    def __init__(self, num_classes=5, use_cbam=False, input_channels=12):
        super().__init__()
        if input_channels != 12:
            raise ValueError('PTB-XL Wavelet ablation requires 12 ECG channels')
        block = CBAMResBlock if use_cbam else ResBlock
        block_args = dict(use_cbam=True) if use_cbam else {}
        backbone = XResNet1d(
            block, 4, [3, 4, 23, 3], input_channels=input_channels,
            num_classes=num_classes, kernel_size=5, ps_head=0.5,
            lin_ftrs_head=[128], **block_args
        )
        self.ecg_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.ecg_embedding_dim = 512
        self.use_wavelet = False
        self.use_cbam = use_cbam
        self.classifier = nn.Linear(self.ecg_embedding_dim, num_classes)

    def encode_ecg(self, ecg):
        if ecg.dim() != 3 or ecg.shape[1] != 12 or ecg.shape[2] != 1000:
            raise ValueError('ECG input must have shape [B, 12, 1000]')
        feature_map = self.ecg_backbone(ecg)
        return torch.cat((feature_map.max(dim=2).values, feature_map.mean(dim=2)), dim=1)

    def forward(self, ecg):
        return self.classifier(self.encode_ecg(ecg))


def build_wavelet_ablation_model(num_classes=5, use_cbam=False, feature_type=None,
                                 fusion_type='none', feature_shape=None,
                                 feature_hidden_dim=256, feature_embedding_dim=128,
                                 feature_dropout=0.2, fusion_hidden_dim=256,
                                 fusion_dropout=0.2):
    if feature_type is None:
        if fusion_type != 'none':
            raise ValueError('ECG-only models require fusion_type="none"')
        return ECGOnlyXResNet(num_classes=num_classes, use_cbam=use_cbam)
    if feature_type != 'wavelet' or fusion_type != 'concat' or tuple(feature_shape or ()) != (12, 6):
        raise ValueError('Wavelet late fusion requires feature_type="wavelet", feature_shape=(12, 6), fusion_type="concat"')
    return WaveletLateFusionXResNet(
        num_classes=num_classes, use_cbam=use_cbam,
        feature_hidden_dim=feature_hidden_dim,
        feature_embedding_dim=feature_embedding_dim,
        feature_dropout=feature_dropout,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
    )
