from utils import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
import json
from itertools import repeat
from pathlib import Path
from utils import emd_features

SUPERCLASS_LABELS = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat', database_filename=None, dataset_type=None, skip_training=False):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency
        self.database_filename = database_filename
        self.dataset_type = dataset_type
        self.skip_training = skip_training

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def prepare(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency, database_filename=self.database_filename, dataset_type=self.dataset_type)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        class_order = SUPERCLASS_LABELS if self.task == 'superdiagnostic' else None
        self.data, self.labels, self.Y, self.mlb = utils.select_data(
            self.data, self.labels, self.task, self.min_samples,
            self.outputfolder+self.experiment_name+'/data/', class_order=class_order
        )
        print('Task: {}, classes: {}, n_classes: {}'.format(self.task, self.mlb.classes_.tolist(), self.Y.shape[1]))
        if self.task == 'superdiagnostic':
            expected = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
            if self.mlb.classes_.tolist() != expected:
                raise ValueError('Superdiagnostic class order mismatch: {}'.format(self.mlb.classes_.tolist()))
            if self.Y.shape[1] != 5:
                raise ValueError('Superdiagnostic must have 5 classes, got {}'.format(self.Y.shape[1]))
        self.emd_data = None
        self.emd_feature_columns = None
        emd_config = self._get_emd_config()
        if emd_config is not None:
            self._prepare_emd_features(emd_config)
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        if self.emd_data is not None:
            self.emd_train = self.emd_data[self.labels.strat_fold <= self.train_fold]
            self.emd_val = self.emd_data[self.labels.strat_fold == self.val_fold]
            self.emd_test = self.emd_data[self.labels.strat_fold == self.test_fold]
            print('EMD train/val/test counts: {}/{}/{}'.format(
                len(self.emd_train), len(self.emd_val), len(self.emd_test)
            ))

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        if self.emd_data is not None:
            mean, std = emd_features.fit_emd_standardizer(self.emd_train)
            self.emd_train = emd_features.apply_emd_standardizer(self.emd_train, mean, std)
            self.emd_val = emd_features.apply_emd_standardizer(self.emd_val, mean, std)
            self.emd_test = emd_features.apply_emd_standardizer(self.emd_test, mean, std)
            data_path = Path(self.outputfolder+self.experiment_name+'/data/')
            emd_features.save_emd_standardizer(data_path/'emd_scaler.npz', mean, std, self.emd_feature_columns)
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')

    def perform(self):

        for model_description in self.models:
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]
            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "cbam_xresnet1d_model":
                from models.cbam_xresnet1d_model import cbam_xresnet1d_model
                if self.emd_data is None and modelparams.get('input_mode', 'late_fusion') != 'ecg_only':
                    raise ValueError('EMD features are required for {} mode'.format(modelparams.get('input_mode')))
                modelparams = self._network_model_params(modelparams)
                model = cbam_xresnet1d_model(modelname, n_classes, self.sampling_frequency, mpath,
                                              self.input_shape, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            if self.skip_training and modeltype == "fastai_model":
                self._predict_with_pretrained(model, modelname, mpath)
            elif modeltype == "cbam_xresnet1d_model":
                if self.emd_data is None:
                    X_train, X_val, X_test = self.X_train, self.X_val, self.X_test
                else:
                    X_train = self._paired_inputs(self.X_train, self.emd_train)
                    X_val = self._paired_inputs(self.X_val, self.emd_val)
                    X_test = self._paired_inputs(self.X_test, self.emd_test)
                model.fit(X_train, self.y_train, X_val, self.y_val)
                model.predict(X_train).dump(mpath+'y_train_pred.npy')
                model.predict(X_val).dump(mpath+'y_val_pred.npy')
                model.predict(X_test).dump(mpath+'y_test_pred.npy')
            else:
                # fit model
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
                # predict and dump
                model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
                model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
                model.predict(self.X_test).dump(mpath+'y_test_pred.npy')

        modelname = 'ensemble'
        # create ensemble predictions via simple mean across model predictions (except naive predictions)
        ensemblepath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(ensemblepath):
            os.makedirs(ensemblepath)
        if not os.path.exists(ensemblepath+'results/'):
            os.makedirs(ensemblepath+'results/')
        # load all predictions
        ensemble_train, ensemble_val, ensemble_test = [],[],[]
        for model_description in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
            if not model_description in ['ensemble', 'naive']:
                mpath = self.outputfolder+self.experiment_name+'/models/'+model_description+'/'
                ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')

    def _get_emd_config(self):
        configs = []
        for model in self.models:
            if model['modeltype'] == 'cbam_xresnet1d_model' and model['parameters'].get('input_mode', 'late_fusion') != 'ecg_only':
                configs.append(model['parameters'])
        if not configs:
            return None
        config = configs[0]
        paths = config.get('emd_feature_paths')
        scenario = config.get('emd_scenario')
        if not paths or not scenario:
            raise ValueError('EMD experiments require emd_feature_paths and emd_scenario')
        if scenario not in paths:
            raise ValueError('EMD scenario {} is not present in emd_feature_paths'.format(scenario))
        for other in configs[1:]:
            if other.get('emd_scenario') != scenario or other.get('emd_feature_paths') != paths:
                raise ValueError('All EMD models in one experiment must use the same EMD source')
        waveform_scenario = config.get('waveform_scenario', scenario)
        if waveform_scenario != scenario:
            raise ValueError('waveform_scenario and emd_scenario must match')
        return config

    def _prepare_emd_features(self, config):
        paths = config['emd_feature_paths']
        scenario = config['emd_scenario']
        all_paths = config.get('emd_feature_paths_for_schema', list(paths.values()))
        columns = emd_features.resolve_emd_feature_columns(all_paths, config.get('emd_feature_columns'))
        record_ids, features, incomplete = emd_features.load_emd_features(
            paths[scenario], self.labels, columns, self.labels.index.values,
            config.get('missing_record_policy', 'drop'), config.get('feature_log_transform', False),
            config.get('log_feature_columns')
        )
        positions = self.labels.index.get_indexer(record_ids)
        if (positions < 0).any():
            raise ValueError('EMD record IDs could not be aligned to selected metadata')
        self.data = self.data[positions]
        self.labels = self.labels.loc[record_ids]
        self.Y = self.Y[positions]
        self.emd_data = features
        self.emd_feature_columns = columns
        data_path = Path(self.outputfolder+self.experiment_name+'/data/')
        data_path.mkdir(parents=True, exist_ok=True)
        with open(data_path/'emd_config.json', 'w') as file:
            json.dump({
                'emd_scenario': scenario,
                'waveform_scenario': config.get('waveform_scenario', scenario),
                'emd_feature_file': str(paths[scenario]),
                'feature_columns': columns,
                'task': self.task,
                'num_classes': int(self.Y.shape[1]),
                'class_names': self.mlb.classes_.tolist(),
                'number_of_emd_features': len(columns),
                'number_of_dropped_records': len(incomplete),
                'dropped_record_ids': [int(record_id) for record_id in incomplete],
            }, file, indent=2)
        print('EMD scenario: {}, file: {}'.format(scenario, paths[scenario]))
        print('EMD shape: {}, dropped records: {}'.format(features.shape, len(incomplete)))

    def _network_model_params(self, params):
        excluded = {
            'emd_feature_paths', 'emd_feature_paths_for_schema', 'emd_scenario', 'waveform_scenario',
            'emd_feature_columns', 'missing_record_policy', 'feature_log_transform', 'log_feature_columns'
        }
        return {key: value for key, value in params.items() if key not in excluded}

    def _paired_inputs(self, X, emd):
        if len(X) != len(emd):
            raise AssertionError('ECG and EMD sample counts differ')
        return list(zip(X, emd))

    def _predict_with_pretrained(self, model, modelname, mpath):
        import torch
        weight_path = Path(mpath) / 'models' / (modelname + '.pth')
        if not weight_path.exists():
            print(f"Pre-trained weight not found at {weight_path}, training from scratch...")
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        else:
            print(f"Loading pre-trained weights from {weight_path}...")
            checkpoint = torch.load(str(weight_path), map_location='cpu', weights_only=False)
            state = checkpoint.get('model', checkpoint)
            pretrained_classes = None
            for key in reversed(list(state.keys())):
                if hasattr(state[key], 'shape'):
                    pretrained_classes = state[key].shape[0]
                    break

            current_classes = self.Y.shape[1]
            if pretrained_classes and pretrained_classes != current_classes:
                print(f"Warning: Pre-trained classes ({pretrained_classes}) != current classes ({current_classes}). "
                      f"Using pre-trained class count for predictions.")
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, pretrained_classes, self.sampling_frequency, mpath,
                                     self.input_shape, **model.__dict__)

        model.predict(self.X_train).dump(mpath + 'y_train_pred.npy')
        model.predict(self.X_val).dump(mpath + 'y_val_pred.npy')
        model.predict(self.X_test).dump(mpath + 'y_test_pred.npy')
        print(f"Predictions saved for {modelname}")

    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):

        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        #y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                #train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                #val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            #train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            #val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        #train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        #val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            print(m)
            mpath = self.outputfolder+self.experiment_name+'/models/'+m+'/'
            rpath = self.outputfolder+self.experiment_name+'/models/'+m+'/results/'

            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            #y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            if self.experiment_name == 'exp_ICBEB':
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)
            else:
                thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)
            te_df_result = pd.DataFrame(
                np.array([
                    te_df_point.mean().values, 
                    te_df.mean().values,
                    te_df.quantile(0.05).values,
                    te_df.quantile(0.95).values]), 
                columns=te_df.columns, 
                index=['point', 'mean', 'lower', 'upper'])

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')
