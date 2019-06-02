import h5py
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import concatenate_sets, \
    split_into_train_test
import numpy as np
import os
from src.base.base_data_loader import BaseDataLoader


def get_dataset(list_subject_ids, i_valid_fold, config):
    if config['experiment']['dataset'] == 'bciciv2a':
        return BCICIV2aProcessed(list_subject_ids, i_valid_fold)
    elif config['experiment']['dataset'] == 'hgd':
        return HighGammaProcessed(list_subject_ids, i_valid_fold)


class BCICIV2aProcessed(BaseDataLoader):
    def __init__(self, list_subject_ids, i_valid_fold):
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..',
                         'data/bciciv2a_processed_low_cut_4hz'))
        train_set, valid_set, test_set = load_h5_data_and_split(self.data_dir,
                                                                list_subject_ids,
                                                                i_valid_fold)
        super().__init__(train_set, valid_set, test_set, 4)


class HighGammaProcessed(BaseDataLoader):
    def __init__(self, list_subject_ids, i_valid_fold):
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..',
                         'data/hgd_processed_low_cut_4hz'))
        train_set, valid_set, test_set = load_h5_data_and_split(self.data_dir,
                                                                list_subject_ids,
                                                                i_valid_fold)
        super().__init__(train_set, valid_set, test_set, 4)


def load_h5_data_and_split(data_dir, list_subject_ids, i_valid_fold):
    """
    Loads data in format <id>_[train/test].h5
    and splits train into train, validation sets
    :return: train, validation and test sets
    """
    # Load sets from h5 files:
    full_train_set = _load_and_merge_data(
        [f"{data_dir}/{i}_train.h5" for i in list_subject_ids])
    test_set = _load_and_merge_data(
        [f"{data_dir}/{i}_test.h5" for i in list_subject_ids])

    # Shuffle train set if multiple subjects merged
    if len(list_subject_ids) > 1:
        full_train_set = _shuffle_signal_and_target(full_train_set)

    # Split into train and valid sets
    train_set, valid_set = split_into_train_test(full_train_set, 4,
                                                 i_valid_fold)

    return train_set, valid_set, test_set


def _load_and_merge_data(file_paths):
    """
    Load multiple HGD subjects and merged them into a new dataset
    :param file_paths:
    :return:
    """
    signal_and_target_data_list = []
    for path in file_paths:
        signal_and_target_data_list.append(_load_h5_data(path))
    if len(file_paths) == 1:
        return signal_and_target_data_list[0]
    return concatenate_sets(signal_and_target_data_list)


def _load_h5_data(file_path):
    """
    Loads HGD h5 file data and create SignalAndTarget object
    :param file_path:
    :return:
    """
    with h5py.File(file_path, 'r') as h5file:
        keys = sorted(list(h5file.keys()))  # 0 is X, 1 is y
        return SignalAndTarget(h5file[keys[0]][()], h5file[keys[1]][()])


def _shuffle_signal_and_target(full_train_set):
    np.random.seed(0)
    indices = np.arange(full_train_set.y.shape[0])
    np.random.shuffle(indices)
    full_train_set.X = full_train_set.X[indices]
    full_train_set.y = full_train_set.y[indices]
    return full_train_set
