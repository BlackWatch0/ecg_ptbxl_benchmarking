import sys
sys.path.insert(0, '.')

import torch
from fastai.basic_train import Learner

# Monkey-patch Learner.load to use weights_only=False (compatibility fix for PyTorch >= 2.6)
_original_learner_load = Learner.load

def _patched_load(self, name_or_path, with_opt=False, device='cpu'):
    from pathlib import Path
    if isinstance(name_or_path, str) and not name_or_path.endswith('.pth'):
        source = self.path / self.model_dir / f'{name_or_path}.pth'
    else:
        source = Path(name_or_path)
    state = torch.load(str(source), map_location=device, weights_only=False)
    if set(state.keys()) == {'model', 'opt'}:
        if with_opt:
            self.opt.load_state_dict(state['opt'])
        self.model.load_state_dict(state['model'])
    else:
        self.model.load_state_dict(state)

Learner.load = _patched_load

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101

datafolder = '../data/ptbxl_clean_no_noise/'
outputfolder = '../output/'
database_filename = 'ptbxl_database_clean_no_noise.csv'
dataset_type = 'ptbxl'
task = 'all'

models = [conf_fastai_xresnet1d101]

e = SCP_Experiment(
    'exp0', task, datafolder, outputfolder, models,
    database_filename=database_filename,
    dataset_type=dataset_type,
    skip_training=True,
)

print("Preparing data...")
e.prepare()
print(f"Data prepared: train={len(e.X_train)}, val={len(e.X_val)}, test={len(e.X_test)}, classes={e.n_classes}")

print("Running inference (skip_training=True)...")
e.perform()

print("Evaluating results...")
e.evaluate(bootstrap_eval=False)

print("\n=== Results ===")
import pandas as pd
for m in ['naive', 'fastai_xresnet1d101']:
    rpath = f'../output/exp0/models/{m}/results/'
    df = pd.read_csv(rpath + 'te_results.csv', index_col=0)
    print(f"{m}: macro_auc = {df.loc['point', 'macro_auc']:.4f}")
