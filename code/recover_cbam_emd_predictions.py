from copy import deepcopy
from pathlib import Path
import os

from experiments.scp_experiment import SCP_Experiment
from configs.cbam_configs import conf_cbam_xresnet1d101_late_fusion_superdiagnostic
from models.cbam_xresnet1d_model import cbam_xresnet1d_model


datafolder = '../data/ptbxl_clean_no_noise/'
outputfolder = '../output/'
experiment_name = 'exp_emd_late_fusion_superdiagnostic'

config = deepcopy(conf_cbam_xresnet1d101_late_fusion_superdiagnostic)
config['parameters']['input_size'] = float(os.environ.get('CBAM_INPUT_SIZE', config['parameters']['input_size']))
config['parameters']['chunkify_train'] = False
config['parameters']['chunkify_valid'] = config['parameters']['input_size'] != 10.0

experiment = SCP_Experiment(
    experiment_name, 'superdiagnostic', datafolder, outputfolder, [config],
    database_filename='ptbxl_database_clean_no_noise.csv', dataset_type='ptbxl'
)
experiment.prepare()

modelname = config['modelname']
mpath = outputfolder + experiment_name + '/models/' + modelname + '/'
params = experiment._network_model_params(config['parameters'])
model = cbam_xresnet1d_model(modelname, experiment.n_classes, experiment.sampling_frequency,
                              mpath, experiment.input_shape, **params)
if not (Path(mpath) / 'models' / '{}.pth'.format(modelname)).exists():
    model.name = 'best_valid_loss'

X_train = experiment._paired_inputs(experiment.X_train, experiment.emd_train)
X_val = experiment._paired_inputs(experiment.X_val, experiment.emd_val)
X_test = experiment._paired_inputs(experiment.X_test, experiment.emd_test)
model.predict(X_train).dump(mpath + 'y_train_pred.npy')
model.predict(X_val).dump(mpath + 'y_val_pred.npy')
model.predict(X_test).dump(mpath + 'y_test_pred.npy')
model.predict_logits(X_val).dump(mpath + 'y_val_logits.npy')
experiment.evaluate(bootstrap_eval=False)
