import h5py
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import concatenate_sets, \
    split_into_train_test
import numpy as np
from base.base_data_loader import BaseDataLoader
from data_loader.util.pair_data import create_paired_dataset, \
    split_paired_into_train_test
import random


def get_dataset(subject_id, i_valid_fold, dataset_name, config):
    data_dir = config['data'][dataset_name]['proc_path']
    n_classes = config['data'][dataset_name]['n_classes']
    n_subjects = config['data'][dataset_name]['n_subjects']
    n_folds = config['experiment']['n_folds']

    if config['experiment']['type'] == 'ccsa_da':
        source_subject_ids = [i for i in range(1, n_subjects + 1) if
                              i != subject_id]
        return PairedProcessedData(subject_id, source_subject_ids, n_folds,
                                   i_valid_fold,
                                   data_dir, n_classes)
    else:
        return ProcessedDataset(subject_id, n_folds, i_valid_fold, data_dir,
                                n_classes)


class PairedProcessedData(BaseDataLoader):
    def __init__(self, target_subject_id, list_source_subject_ids, n_folds,
                 i_valid_fold, data_dir, n_classes):
        # Load data from source and target
        tgt_full_train_set, tgt_test_set = _load_data(data_dir,
                                                      target_subject_id)

        src_full_train_set, src_test_set = _load_data(data_dir,
                                                      list_source_subject_ids)
        # Create paired dataset
        paired_full_train_set = create_paired_dataset(tgt_full_train_set,
                                                      src_full_train_set,
                                                      n_classes)
        # Split into train and valid sets
        train_set, valid_set = split_paired_into_train_test(
            paired_full_train_set, n_folds,
            i_valid_fold, n_classes)
        super().__init__(train_set, valid_set, None, n_classes)


class ProcessedDataset(BaseDataLoader):
    def __init__(self, subject_ids, n_folds, i_valid_fold, data_dir,
                 n_classes):
        # Load data
        full_train_set, test_set = _load_data(data_dir,
                                              subject_ids)
        # Split into train and valid sets
        train_set, valid_set = split_into_train_test(full_train_set, n_folds,
                                                     i_valid_fold)
        super().__init__(train_set, valid_set, test_set, n_classes)


def _load_data(data_dir, subject_ids):
    assert type(subject_ids) is list or type(subject_ids) is int, \
        "Subject ids must be list or int (for single subject)"
    # Make list if single subject
    if type(subject_ids) is int:
        subject_ids = [subject_ids]

    # Load sets from h5 files:
    full_train_set = _load_and_merge_data(
        [f"{data_dir}/{i}_train.h5" for i in subject_ids])
    test_set = _load_and_merge_data(
        [f"{data_dir}/{i}_test.h5" for i in subject_ids])

    # Shuffle train set if multiple subjects merged
    if len(subject_ids) > 1:
        full_train_set = _shuffle_signal_and_target(full_train_set)

    # back to numpy array:
    full_train_set.X = np.array(full_train_set.X)
    full_train_set.y = np.array(full_train_set.y)
    test_set.X = np.array(test_set.X)
    test_set.y = np.array(test_set.y)

    return full_train_set, test_set


def _load_and_merge_data(file_paths):
    """
    Load multiple HGD subjects and merged them into a new dataset
    :param file_paths:
    :return:
    """
    if len(file_paths) == 1:  # if just 1 subject's data
        return _load_h5_data(file_paths[0])

    signal_and_target_data_list = []
    for path in file_paths:
        signal_and_target_data_list.append(_load_h5_data(path))

    return concatenate_sets(signal_and_target_data_list)


def _load_h5_data(file_path):
    """
    Loads HGD h5 file data and create SignalAndTarget object
    :param file_path:
    :return:
    """
    with h5py.File(file_path, 'r') as h5file:
        keys = sorted(list(h5file.keys()))  # 0 is X, 1 is y
        # convert to list for faster indexing later on
        a=h5file[keys[0]][()]
        b=h5file[keys[1]][()]
        c=list(h5file[keys[0]][()])
        d=list(h5file[keys[1]][()])
        return SignalAndTarget(
            list(h5file[keys[0]][()]),
            list(h5file[keys[1]][()])
        )


def _shuffle_signal_and_target(full_train_set):
    random.seed(0)
    indices = [x for x in range(len(full_train_set.y))]
    random.shuffle(indices)
    full_train_set.X = full_train_set.X[indices]
    full_train_set.y = full_train_set.y[indices]
    return full_train_set
