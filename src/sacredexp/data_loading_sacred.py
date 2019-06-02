import pickle
import os
from collections import OrderedDict

import braindecode.datautil.splitters as splitters
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import split_into_two_sets

from sacred import Ingredient

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    dataset_name = "bcic_iv_2a"
    raw_data_path = "/Users/../../..gdf&mat"  # path to original raw gdf and mat files from BCIC IV 2a
    pickle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                               'data')) + "/bcic_iv_2a_all_9_subjects.pickle"  # path to pickle dir

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
    # train:[1,2,..,6] valid:[7,8] => models, warmstart with models on [9] (train, valid, test)
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
def load_bcic_iv_2a_data(raw_data_path, pickle_path, from_pickle, data_preprocessing, subject_ids):
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
