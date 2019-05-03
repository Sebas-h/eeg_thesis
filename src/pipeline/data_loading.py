import pickle
import os
from collections import OrderedDict
import glob

import braindecode.datautil.splitters as splitters
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import split_into_two_sets

import logging
import numpy as np
from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
import h5py
import scipy.io as sio

from sacred import Ingredient

data_ingredient = Ingredient('dataset')

log = logging.getLogger(__name__)
log.setLevel('INFO')


def main():
    load_high_gamma_dataset()


def pickle_to_mat():
    import scipy.io as sio
    dire = "/Users/sebas/Downloads/"

    data = load_bcic_iv_2a_data(True, 'all')

    for idx, s in enumerate(data):
        sio.savemat(dire + f"subject_{idx + 1}.mat", {'X': s.X, 'y': s.y})

    # mm = sio.loadmat('tests1mat.mat')
    # print(mm.items())
    print('done')


@data_ingredient.config
def cfg():
    dataset_name = "bcic_iv_2a"
    raw_data_path = "/Users/../../..gdf&mat"  # path to original raw gdf and mat files from BCIC IV 2a
    pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pickle_path = pickle_dir + "/bcic_iv_2a_all_9_subjects.pickle"  # path to pickle dir

    n_classes = 4  # number of classes in dataset
    subject_ids = [1]  # 1-9, list of ids or 'all' for all subjects
    n_folds = 4  # one fold validation, one fold test, the rest is train fold(s)

    from_pickle = True  # if not from_pickle then data from raw with preprocessing
    # data preprocessing parameters
    data_preprocessing = {
        'ival': [-500, 4000],
        'low_cut_hz': 4,
        'high_cut_hz': 38,
        'factor_new': 1e-3,
        'init_block_size': 1000
    }


@data_ingredient.capture
def get_data_tl(n_folds, cv=False):
    # train:[1,2,..,6] valid:[7,8] => model, warmstart with model on [9] (train, valid, test)
    # [ [train_abo, valid_abo], [train, valid, test] ] , [ [...], [...] ], ..., [       ]
    # return :=
    #  [
    #   [ [train_abo, valid_abo], [train, valid, test], [subject_id, subject_ids_others] ] ,
    #   [ [...], [...] ],
    #   ...,
    #   [       ]
    #  ]

    result_datasets = []
    data_all = load_bcic_iv_2a_data()
    for i in range(len(data_all)):
        subject_id = i + 1
        subject_ids = list(filter(lambda x: x != subject_id, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
        data_one = data_all[i]
        data_rest = []
        for si in subject_ids:
            data_rest.append(data_all[si - 1])
        train_abo = splitters.concatenate_sets(data_rest[:-2])
        valid_abo = splitters.concatenate_sets(data_rest[-2:])
        train, valid, test = splitters.split_into_train_valid_test(data_one, n_folds, 0)
        result_datasets.append(
            [
                [train_abo, valid_abo],
                [train, valid, test],
                [subject_id, subject_ids]
            ]
        )
    return result_datasets


@data_ingredient.capture
def get_data(n_folds, cv=False):
    if cv:
        cv_folds = n_folds - 1
    else:
        cv_folds = 1
    subjects_data = load_bcic_iv_2a_data()
    res_datasets = []
    for data in subjects_data:
        train_valid_set, test_set = splitters.split_into_train_test(data, n_folds, 0)
        # Cross validation folds (train and valid folds)
        cv_fold_sets = []
        for i in range(cv_folds):
            cv_fold_sets.append(splitters.split_into_train_test(train_valid_set, n_folds - 1, i))
        res_datasets.append([cv_fold_sets, test_set])
    return res_datasets


@data_ingredient.capture
# def load_bcic_iv_2a_data(raw_data_path, pickle_path, from_pickle, data_preprocessing, subject_ids):
def load_bcic_iv_2a_data(from_pickle, subject_ids):
    ##########################################################
    ##########################################################
    # cfg
    # path to original raw gdf and mat files from BCIC IV 2a:
    raw_data_path = "/Users/sebas/code/_eeg_data/BCICIV_2a_gdf"
    # path to pickle dir:
    pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
    # full path:
    pickle_path = pickle_dir + "/bcic_iv_2a_all_9_subjects.pickle"

    data_preprocessing = {
        'ival': [-500, 4000],
        'low_cut_hz': 4,
        'high_cut_hz': 38,
        'factor_new': 1e-3,
        'init_block_size': 1000
    }
    ##########################################################
    ##########################################################

    if from_pickle and pickle_path is not None:

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        if subject_ids != 'all':
            r_data = []
            for subject_id in subject_ids:
                r_data.append(data[subject_id - 1])
            data = r_data
    else:
        ival = data_preprocessing['ival']
        low_cut_hz = data_preprocessing['low_cut_hz']
        high_cut_hz = data_preprocessing['high_cut_hz']
        factor_new = data_preprocessing['factor_new']
        init_block_size = data_preprocessing['init_block_size']

        if subject_ids == 'all':
            subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        data = []

        for subject_id in subject_ids:
            train_filename = 'A{:02d}T.gdf'.format(subject_id)
            test_filename = 'A{:02d}E.gdf'.format(subject_id)
            train_filepath = os.path.join(raw_data_path, train_filename)
            test_filepath = os.path.join(raw_data_path, test_filename)
            train_label_filepath = train_filepath.replace('.gdf', '.mat')
            test_label_filepath = test_filepath.replace('.gdf', '.mat')

            train_loader = BCICompetition4Set2A(
                train_filepath, labels_filename=train_label_filepath)
            test_loader = BCICompetition4Set2A(
                test_filepath, labels_filename=test_label_filepath)
            train_cnt = train_loader.load()
            test_cnt = test_loader.load()

            # Preprocessing
            train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                                 'EOG-central', 'EOG-right'])
            assert len(train_cnt.ch_names) == 22
            # lets convert to millvolt for numerical stability of next operations
            train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
            train_cnt = mne_apply(
                lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                                       filt_order=3,
                                       axis=1), train_cnt)
            train_cnt = mne_apply(
                lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                          init_block_size=init_block_size,
                                                          eps=1e-4).T,
                train_cnt)

            test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                               'EOG-central', 'EOG-right'])
            assert len(test_cnt.ch_names) == 22
            test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
            test_cnt = mne_apply(
                lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                                       filt_order=3,
                                       axis=1), test_cnt)
            test_cnt = mne_apply(
                lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                          init_block_size=init_block_size,
                                                          eps=1e-4).T, test_cnt)

            marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                                      ('Foot', [3]), ('Tongue', [4])])

            train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
            test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
            data_subject = splitters.concatenate_two_sets(train_set, test_set)
            data.append(data_subject)

    return data


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                         clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_high_gamma_dataset():
    low_cut_hz = 0  # 0 or 4 Hz
    debug = True
    test_set_path = '/Users/sebas/code/_eeg_data/high-gamma-dataset/data/test'
    train_set_path = '/Users/sebas/code/_eeg_data/high-gamma-dataset/data/train'

    test_set_mat_files = glob.glob(test_set_path + '/*.mat')
    train_set_mat_files = glob.glob(train_set_path + '/*.mat')
    test_filename = test_set_mat_files[0]
    train_filename = train_set_mat_files[0]

    log.info("Loading train...")
    full_train_set = load_bbci_data(train_filename, low_cut_hz=low_cut_hz, debug=debug)
    log.info("Loading test...")
    test_set = load_bbci_data(test_filename, low_cut_hz=low_cut_hz, debug=debug)
    print('done')


def load_high_gamma_parameters():
    from collections import OrderedDict
    import torch
    from braindecode.models.deep4 import Deep4Net

    model_type = 'shallow'  # 'deep' or 'shallow' weights are available
    path = '/Users/sebas/code/_eeg_data/high-gamma-dataset/data/trained-parameters/' + model_type
    param_pickles = glob.glob(path + "/*.pkl")
    file_name = param_pickles[0]
    # file_name = "/Users/sebas/code/thesis/pipeline/1.pkl"

    model_params = torch.load(file_name, map_location='cpu')
    # model_iv = torch.load('/Users/sebas/code/thesis/pipeline/model_sate_s2_deep.pt', map_location='cpu')
    dataset_bcic_iv_2a = load_bcic_iv_2a_data(from_pickle=True, subject_ids='all')
    # data_subject_1 = dataset_bcic_iv_2a[0]
    data_subject_2 = dataset_bcic_iv_2a[1]
    inputs = data_subject_2.X
    targets = data_subject_2.y

    model = Deep4Net(22, 4, input_time_length=1125, final_conv_length='auto').create_network()
    model.load_state_dict(model_params)
    model.eval()
    outputs = model(inputs)
    pred_labels = np.argmax(outputs, axis=1).squeeze()
    accuracy = np.mean(pred_labels == targets)
    print(accuracy)
    print('done')


if __name__ == '__main__':
    main()
