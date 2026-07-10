import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

###############################################################################
# Match the Lightning checkpoint architecture
###############################################################################

class ConvBlock1d(nn.Sequential):
    """Conv1d -> BN1d -> ReLU (matching lightning ConvLayer structure)"""
    def __init__(self, ni, nf, ks=3, stride=1, bias=True):
        conv = nn.Conv1d(ni, nf, ks, stride=stride, padding=(ks-1)//2, bias=bias)
        bn = nn.BatchNorm1d(nf)
        act = nn.ReLU(inplace=True)
        super().__init__(conv, bn, act)

class BottleneckResBlock(nn.Module):
    """bottleneck ResBlock (expansion=4) matching lightning structure"""
    def __init__(self, ni, nf, stride=1, kernel_size=5):
        super().__init__()
        expansion = 4
        ni_exp = ni * expansion
        nf_exp = nf * expansion
        nh1 = nf  # intermediate channels
        nh2 = nf
        
        # concs: 1x1 reduce -> 3x3 conv -> 1x1 expand
        layers = nn.Sequential(
            ConvBlock1d(ni_exp, nh1, ks=1),
            ConvBlock1d(nh1, nh2, ks=kernel_size, stride=stride),
            nn.Sequential(
                nn.Conv1d(nh2, nf_exp, 1, bias=True),
                nn.BatchNorm1d(nf_exp),
            )
        )
        self.convs = layers
        
        convpath_layers = [self.convs]  # convpath[0] = convs
        self.convpath = nn.Sequential(*convpath_layers)
        
        # idpath
        id_layers = []
        if ni_exp != nf_exp or stride != 1:
            id_layers.append(nn.Sequential(
                nn.Conv1d(ni_exp, nf_exp, 1, stride=stride, bias=True),
                nn.BatchNorm1d(nf_exp),
            ))
        self.idpath = nn.Sequential(*id_layers)
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.convpath(x)
        if len(self.idpath) > 0:
            out = out + self.idpath(x)
        else:
            out = out + x
        return self.act(out)


# The first block group has stride=1, others stride=2
# For first block in a group (i==0), there's a stride and possibly idpath change

def _make_layer(ni, nf, blocks, stride, kernel_size):
    layers = []
    for i in range(blocks):
        layers.append(BottleneckResBlock(
            ni if i == 0 else nf,
            nf,
            stride=stride if i == 0 else 1,
            kernel_size=kernel_size,
        ))
    return nn.Sequential(*layers)


class AdaptiveConcatPool1d(nn.Module):
    """AdaptiveMaxPool1d + AdaptiveAvgPool1d concatenated (doubles channels)"""
    def __init__(self, sz=1):
        super().__init__()
        self.mp = nn.AdaptiveMaxPool1d(sz)
        self.ap = nn.AdaptiveAvgPool1d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class XResNetLightning(nn.Module):
    """Match the Lightning checkpoint model structure"""
    
    def __init__(self, num_classes=71, input_channels=12, layers=[3,4,6,3], 
                 kernel_size=5, expansion=4):
        super().__init__()
        
        # Stem
        stem = nn.Sequential(
            ConvBlock1d(input_channels, 32, ks=kernel_size, stride=2),
            ConvBlock1d(32, 32, ks=kernel_size, stride=1),
            ConvBlock1d(32, 64, ks=kernel_size, stride=1),
        )
        
        # MaxPool
        maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Blocks with expansion=4
        block_szs = [64 // expansion] + [64, 64, 64, 64]
        blocks = []
        for i, l in enumerate(layers):
            ni = block_szs[i]
            nf = block_szs[i+1]
            stride = 1 if i == 0 else 2
            blocks.append(_make_layer(ni, nf, l, stride, kernel_size))
        
        self.features = nn.Sequential(
            *stem,
            maxpool,
            *blocks,
        )
        
        # Concat pooling + BN + classifier
        # concat_pool doubles channels: 256 -> 512
        self.concat_pool = AdaptiveConcatPool1d(1)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Sequential(  # model.1
            nn.Sequential(                # model.1.1
                nn.Linear(512, num_classes),
            )
        )
    
    def forward(self, x):
        # x: [B, C, T]  (channels-first from ToTensor transform)
        out = self.features(x)          # [B, 256, T']
        out = self.concat_pool(out)     # [B, 512, 1]
        out = out.squeeze(-1)           # [B, 512]
        out = self.bn(out)              # [B, 512]
        out = self.classifier(out)      # [B, num_classes]
        return out


def load_lightning_checkpoint(ckpt_path, num_classes=71, device='cpu'):
    """Load a lightning checkpoint and return the model"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['state_dict']
    
    model = XResNetLightning(num_classes=num_classes)
    
    # Map lightning keys to our model keys
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith('model.0.features.'):
            new_key = 'features.' + k[len('model.0.features.'):]
            new_sd[new_key] = v
        elif k == 'model.0.bn.weight':
            new_sd['bn.weight'] = v
        elif k == 'model.0.bn.bias':
            new_sd['bn.bias'] = v
        elif k == 'model.0.bn.running_mean':
            new_sd['bn.running_mean'] = v
        elif k == 'model.0.bn.running_var':
            new_sd['bn.running_var'] = v
        elif k == 'model.0.bn.num_batches_tracked':
            new_sd['bn.num_batches_tracked'] = v
        elif k == 'model.1.1.1.weight':
            new_sd['classifier.0.0.weight'] = v
        elif k == 'model.1.1.1.bias':
            new_sd['classifier.0.0.bias'] = v
    
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    model.to(device)
    return model


###############################################################################
# Inference helpers (no fastai needed)
###############################################################################
from models.timeseries_utils import TimeseriesDatasetCrops, aggregate_predictions

def predict_with_lightning(model, X, input_size=250, batch_size=128, device='cpu'):
    """
    X: list of numpy arrays [T, C] (variable length, channels=12)
    Returns: numpy array [N, num_classes] of predictions (raw logits, not sigmoid)
    """
    import pandas as pd
    num_samples = len(X)
    input_size = int(input_size)  # was 2.5s * 100Hz = 250
    
    # Build chunk dataframe
    df = pd.DataFrame({'data': range(num_samples), 'label': [np.zeros(model.classifier[0][0].out_features) for _ in range(num_samples)]})
    df['data_length'] = [len(x) for x in X]
    
    # Use npy mode with pre-loaded data
    X_npy = [x.astype(np.float32) for x in X]
    
    from models.timeseries_utils import ToTensor, CenterCrop
    tfms = [ToTensor()]
    
    ds = TimeseriesDatasetCrops(
        df, input_size, num_classes=model.classifier[0][0].out_features,
        chunk_length=input_size, min_chunk_length=input_size,
        stride=input_size // 2,
        transforms=tfms, annotation=False,
        col_lbl='label', npy_data=X_npy,
    )
    
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            data, _ = batch
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    # Sigmoid for multi-label
    preds = 1.0 / (1.0 + np.exp(-preds))
    
    idmap = ds.get_id_mapping()
    return aggregate_predictions(preds, idmap=idmap, aggregate_fn=np.amax)


if __name__ == '__main__':
    import pandas as pd
    
    # Load data (same as SCP_Experiment.prepare)
    from utils import utils
    
    datafolder = '../data/ptbxl_clean_no_noise/'
    outputfolder = '../output/'
    task = 'all'
    
    print("Loading dataset...")
    data, raw_labels = utils.load_dataset(datafolder, 100, 
                                          database_filename='ptbxl_database_clean_no_noise.csv',
                                          dataset_type='ptbxl')
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
    data, labels, Y, mlb = utils.select_data(data, labels, task, 0, outputfolder + 'exp0/data/')
    
    # Load saved standardizer
    import pickle
    ss = pickle.load(open(outputfolder + 'exp0/data/standard_scaler.pkl', 'rb'))
    X_std = utils.apply_standardizer(data, ss)
    
    # Split test
    test_fold = 10
    X_test_list = [X_std[i] for i in range(len(X_std)) if labels.strat_fold.iloc[i] == test_fold]
    y_test = Y[labels.strat_fold == test_fold]
    
    print(f"Test: {len(X_test_list)} samples, {y_test.shape[1]} classes")
    
    # Test each available lightning model
    import os
    models_to_test = {
        'xresnet_all': 71,
        'xresnet_superdiagnostic': 5,
        'inception_all': 71,
        'inception_superdiagnostic': 5,
        'resnet_all': 71,
        'resnet_superdiagnostic': 5,
        'lstm_all': 71,
        'lstm_superdiagnostic': 5,
    }
    
    for model_name, n_classes in models_to_test.items():
        ckpt_path = f'../output/{model_name}/checkpoints/best_model.ckpt'
        if not os.path.exists(ckpt_path):
            print(f"\n{model_name}: checkpoint not found, skipping")
            continue
        
        print(f"\n=== {model_name} ({n_classes} classes) ===")
        print(f"Loading checkpoint from {ckpt_path}...")
        
        try:
            model = load_lightning_checkpoint(ckpt_path, num_classes=n_classes)
            
            print("Running inference...")
            y_pred = predict_with_lightning(model, X_test_list, input_size=250)
            
            # Evaluate
            from sklearn.metrics import roc_auc_score
            macro_auc = roc_auc_score(y_test, y_pred, average='macro')
            print(f"macro_auc = {macro_auc:.4f}")
            
            # Save predictions
            save_path = f'../output/{model_name}/y_test_pred_lightning.npy'
            np.save(save_path, y_pred)
            print(f"Predictions saved to {save_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
