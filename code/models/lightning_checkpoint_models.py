import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, ni, nf, ks, stride=1, padding=None):
        if padding is None:
            padding = ks // 2
        super().__init__(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding),
                         nn.BatchNorm1d(nf), nn.ReLU(inplace=True))


def classifier(nf, num_classes):
    return nn.Sequential(nn.Identity(), nn.Sequential(nn.Identity(), nn.Linear(nf, num_classes)))


class LeNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(12, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return F.adaptive_avg_pool1d(self.sequential(x), 1).squeeze(-1)


class LSTMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('0', nn.Identity())
        self.add_module('1', nn.Identity())
        self.add_module('2', nn.LSTM(12, 256, num_layers=2, bidirectional=True))

    def forward(self, x):
        x, _ = self._modules['2'](x.transpose(1, 2).transpose(0, 1))
        x = x.transpose(0, 1).transpose(1, 2)
        avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        last = torch.cat([x[:, :256, -1], x[:, 256:, 0]], 1)
        return torch.cat([avg, maximum, last], 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ni, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(ni, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(12, 64, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(128, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(128, 6, 2)
        self.layer4 = self._make_layer(128, 3, 2)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv1d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)


class BasicConv1d(nn.Module):
    def __init__(self, ni, nf, ks, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = 0
        self.conv = nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(nf, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, ni, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv1d(ni, 64, 1)
        self.branch5x5_1 = BasicConv1d(ni, 48, 1)
        self.branch5x5_2 = BasicConv1d(48, 64, 5, padding=2)
        self.branch3x3dbl_1 = BasicConv1d(ni, 64, 1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, 3, padding=1)
        self.branch_pool = BasicConv1d(ni, pool_features, 1)

    def forward(self, x):
        return torch.cat([self.branch1x1(x), self.branch5x5_2(self.branch5x5_1(x)),
                          self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
                          self.branch_pool(F.avg_pool1d(x, 3, 1, 1))], 1)


class InceptionB(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.branch3x3 = BasicConv1d(ni, 384, 3, stride=2)
        self.branch3x3dbl_1 = BasicConv1d(ni, 64, 1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, 3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch3x3(x), self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
                          F.max_pool1d(x, 3, 2)], 1)


class InceptionC(nn.Module):
    def __init__(self, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv1d(768, 192, 1)
        self.branch7x7_1 = BasicConv1d(768, channels_7x7, 1)
        self.branch7x7_2 = BasicConv1d(channels_7x7, channels_7x7, 7, padding=3)
        self.branch7x7_3 = BasicConv1d(channels_7x7, 192, 7, padding=3)
        self.branch7x7dbl_1 = BasicConv1d(768, channels_7x7, 1)
        self.branch7x7dbl_2 = BasicConv1d(channels_7x7, channels_7x7, 7, padding=3)
        self.branch7x7dbl_3 = BasicConv1d(channels_7x7, channels_7x7, 7, padding=3)
        self.branch7x7dbl_4 = BasicConv1d(channels_7x7, channels_7x7, 7, padding=3)
        self.branch7x7dbl_5 = BasicConv1d(channels_7x7, 192, 7, padding=3)
        self.branch_pool = BasicConv1d(768, 192, 1)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        b3 = self.branch7x7dbl_5(self.branch7x7dbl_4(self.branch7x7dbl_3(self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        return torch.cat([b1, b2, b3, self.branch_pool(F.avg_pool1d(x, 3, 1, 1))], 1)


class InceptionD(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch3x3_1 = BasicConv1d(768, 192, 1)
        self.branch3x3_2 = BasicConv1d(192, 320, 3, stride=2)
        self.branch7x7x3_1 = BasicConv1d(768, 192, 1)
        self.branch7x7x3_2 = BasicConv1d(192, 192, 7, padding=3)
        self.branch7x7x3_3 = BasicConv1d(192, 192, 7, padding=3)
        self.branch7x7x3_4 = BasicConv1d(192, 192, 3, stride=2)

    def forward(self, x):
        b1 = self.branch3x3_2(self.branch3x3_1(x))
        b2 = self.branch7x7x3_4(self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x))))
        return torch.cat([b1, b2, F.max_pool1d(x, 3, 2)], 1)


class InceptionE(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.branch1x1 = BasicConv1d(ni, 320, 1)
        self.branch3x3_1 = BasicConv1d(ni, 384, 1)
        self.branch3x3_2a = BasicConv1d(384, 384, 3, padding=1)
        self.branch3x3_2b = BasicConv1d(384, 384, 3, padding=1)
        self.branch3x3dbl_1 = BasicConv1d(ni, 448, 1)
        self.branch3x3dbl_2 = BasicConv1d(448, 384, 3, padding=1)
        self.branch3x3dbl_3a = BasicConv1d(384, 384, 3, padding=1)
        self.branch3x3dbl_3b = BasicConv1d(384, 384, 3, padding=1)
        self.branch_pool = BasicConv1d(ni, 192, 1)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3_1(x)
        b2 = torch.cat([self.branch3x3_2a(b2), self.branch3x3_2b(b2)], 1)
        b3 = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        b3 = torch.cat([self.branch3x3dbl_3a(b3), self.branch3x3dbl_3b(b3)], 1)
        return torch.cat([b1, b2, b3, self.branch_pool(F.avg_pool1d(x, 3, 1, 1))], 1)


class InceptionAux(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv0 = BasicConv1d(768, 128, 1)
        self.conv1 = BasicConv1d(128, 768, 5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.conv0(F.avg_pool1d(x, 5, stride=3))
        x = self.conv1(x)
        return self.fc(F.adaptive_avg_pool1d(x, 1).squeeze(-1))


class InceptionBackbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv1d(12, 32, 3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv1d(32, 32, 3)
        self.Conv2d_2b_3x3 = BasicConv1d(32, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool1d(3, stride=2, padding=1)
        self.Conv2d_3b_1x1 = BasicConv1d(64, 80, 1)
        self.Conv2d_4a_3x3 = BasicConv1d(80, 192, 3)
        self.maxpool2 = nn.MaxPool1d(3, stride=2, padding=1)
        self.Mixed_5b = InceptionA(192, 32)
        self.Mixed_5c = InceptionA(256, 64)
        self.Mixed_5d = InceptionA(288, 64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(128)
        self.Mixed_6c = InceptionC(160)
        self.Mixed_6d = InceptionC(160)
        self.Mixed_6e = InceptionC(192)
        self.AuxLogits = InceptionAux(num_classes)
        self.Mixed_7a = InceptionD()
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

    def forward(self, x):
        x = self.maxpool1(self.Conv2d_2b_3x3(self.Conv2d_2a_3x3(self.Conv2d_1a_3x3(x))))
        x = self.maxpool2(self.Conv2d_4a_3x3(self.Conv2d_3b_1x1(x)))
        x = self.Mixed_5d(self.Mixed_5c(self.Mixed_5b(x)))
        x = self.Mixed_6e(self.Mixed_6d(self.Mixed_6c(self.Mixed_6b(self.Mixed_6a(x)))))
        x = self.Mixed_7c(self.Mixed_7b(self.Mixed_7a(x)))
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)


class XResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        ni_exp, nf_exp = ni * 4, nf * 4
        self.convs = nn.Sequential(ConvBNReLU(ni_exp, nf, 1), ConvBNReLU(nf, nf, 5, stride),
                                   nn.Sequential(nn.Conv1d(nf, nf_exp, 1), nn.BatchNorm1d(nf_exp)))
        self.convpath = nn.Sequential(self.convs)
        self.idpath = nn.Sequential(*( [nn.Sequential(nn.Conv1d(ni_exp, nf_exp, 1, stride=stride), nn.BatchNorm1d(nf_exp))]
                                       if ni_exp != nf_exp or stride != 1 else [] ))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convpath(x) + (self.idpath(x) if len(self.idpath) else x))


def xres_layer(ni, nf, blocks, stride):
    return nn.Sequential(*[XResBlock(ni if i == 0 else nf, nf, stride if i == 0 else 1) for i in range(blocks)])


class XResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(ConvBNReLU(12, 32, 5, 2), ConvBNReLU(32, 32, 5), ConvBNReLU(32, 64, 5),
                                      nn.MaxPool1d(3, 2, 1), xres_layer(16, 64, 3, 1), xres_layer(64, 64, 4, 1),
                                      xres_layer(64, 64, 6, 1), xres_layer(64, 64, 3, 1))
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([F.adaptive_max_pool1d(x, 1), F.adaptive_avg_pool1d(x, 1)], 1).squeeze(-1)
        return self.bn(x)


class CheckpointModel(nn.Module):
    def __init__(self, backbone, nf, num_classes):
        super().__init__()
        self.model = nn.Sequential(backbone, classifier(nf, num_classes))

    def forward(self, x):
        return self.model[1](self.model[0](x))


def load_checkpoint_model(path, architecture, num_classes, device='cpu'):
    architecture = architecture.lower()
    if architecture == 'lenet':
        model = CheckpointModel(LeNetBackbone(), 128, num_classes)
    elif architecture == 'lstm':
        model = CheckpointModel(LSTMBackbone(), 1536, num_classes)
    elif architecture == 'resnet':
        model = CheckpointModel(ResNetBackbone(), 512, num_classes)
    elif architecture == 'inception':
        model = CheckpointModel(InceptionBackbone(num_classes), 2048, num_classes)
    elif architecture == 'xresnet':
        model = CheckpointModel(XResNetBackbone(), 512, num_classes)
    else:
        raise ValueError('Unsupported architecture: {}'.format(architecture))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint['state_dict']
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError('Checkpoint keys do not match {} architecture'.format(architecture))
    for key, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[key].shape:
            raise RuntimeError('Checkpoint tensor mismatch for {}'.format(key))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
