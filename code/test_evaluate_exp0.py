import sys
sys.path.insert(0, '.')
from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101

e = SCP_Experiment('exp0', 'all', '../data/ptbxl_clean_no_noise/', '../output/', [conf_fastai_xresnet1d101], database_filename='ptbxl_database_clean_no_noise.csv', dataset_type='ptbxl')
e.evaluate(bootstrap_eval=False)

import pandas as pd
for m in ['naive', 'fastai_xresnet1d101']:
    rpath = '../output/exp0/models/' + m + '/results/'
    df = pd.read_csv(rpath + 'te_results.csv', index_col=0)
    print(f"{m}: macro_auc = {df.loc['point', 'macro_auc']}")
