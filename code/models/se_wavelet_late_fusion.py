import torch
import torch.nn as nn

from models.cbam_xresnet1d import CBAMResBlock
from models.xresnet1d import ResBlock, XResNet1d


class WaveletFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

    def forward(self, features):
        if features.dim() != 3 or tuple(features.shape[1:]) != (12, 6):
            raise ValueError('Wavelet features must have shape [B, 12, 6]')
        return self.network(features)


class SEWaveletLateFusionXResNet(nn.Module):
    def __init__(self, num_classes=5, use_se=False, use_wavelet=False):
        super().__init__()
        self.use_se = use_se
        self.use_wavelet = use_wavelet
        block = CBAMResBlock if use_se else ResBlock
        block_args = {'use_cbam': False, 'use_se': True} if use_se else {}
        backbone = XResNet1d(
            block, 4, [3, 4, 23, 3], input_channels=12,
            num_classes=num_classes, kernel_size=5, ps_head=0.5,
            lin_ftrs_head=[128], **block_args
        )
        self.ecg_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.ecg_embedding_dim = 512
        if use_wavelet:
            self.wavelet_encoder = WaveletFeatureEncoder()
            self.fusion = nn.Sequential(
                nn.Linear(self.ecg_embedding_dim + 128, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        else:
            self.wavelet_encoder = None
            self.classifier = nn.Linear(self.ecg_embedding_dim, num_classes)

    def forward_features(self, ecg):
        if ecg.dim() != 3 or tuple(ecg.shape[1:]) != (12, 1000):
            raise ValueError('ECG input must have shape [B, 12, 1000]')
        feature_map = self.ecg_backbone(ecg)
        return torch.cat((feature_map.max(dim=2).values, feature_map.mean(dim=2)), dim=1)

    def forward(self, ecg, features=None):
        embedding = self.forward_features(ecg)
        if not self.use_wavelet:
            return self.classifier(embedding)
        if features is None or features.shape[0] != embedding.shape[0]:
            raise ValueError('Matched Wavelet features are required')
        return self.fusion(torch.cat((embedding, self.wavelet_encoder(features)), dim=1))


def build_model(backbone_name, num_classes, use_se=False, feature_type=None,
                fusion_type='none', feature_shape=None, **kwargs):
    if backbone_name != 'xresnet1d101':
        raise ValueError('Unsupported backbone: {}'.format(backbone_name))
    use_wavelet = feature_type is not None
    if use_wavelet and (feature_type != 'wavelet' or fusion_type != 'concat' or
                        tuple(feature_shape or ()) != (12, 6)):
        raise ValueError('Wavelet late fusion requires wavelet [12, 6] concat features')
    if not use_wavelet and fusion_type != 'none':
        raise ValueError('ECG-only models require fusion_type="none"')
    return SEWaveletLateFusionXResNet(num_classes=num_classes, use_se=use_se,
                                      use_wavelet=use_wavelet)
