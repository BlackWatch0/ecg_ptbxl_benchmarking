from experiments.scp_experiment import SCP_Experiment
from configs.cbam_configs import conf_cbam_xresnet1d101_late_fusion_superdiagnostic


datafolder = '../data/ptbxl_clean_no_noise/'
outputfolder = '../output/'

experiment = SCP_Experiment(
    'exp_emd_late_fusion_superdiagnostic', 'superdiagnostic', datafolder, outputfolder,
    [conf_cbam_xresnet1d101_late_fusion_superdiagnostic],
    database_filename='ptbxl_database_clean_no_noise.csv', dataset_type='ptbxl'
)
experiment.prepare()
experiment.perform()
experiment.evaluate()
