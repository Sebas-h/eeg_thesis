# Unified Deep Supervised Domain Adaptation and Generalization
# by Saeid Motiian et al.

import os
import braindecode.datautil.splitters as data_splitters
from src.pipeline import data_loading
from src.pipeline.run_model import RunModel
import numpy as np

#######################
# CONFIG AND PARAMETERS
#######################

# Data config
dataset_name = "bcic_iv_2a"
raw_data_path = "/Users/../../..gdf&mat"  # path to original raw gdf and mat files from BCIC IV 2a
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
pickle_path = data_dir + "/bcic_iv_2a_all_9_subjects.pickle"  # path to pickle dir
# number of classes in dataset
n_classes = 4
subject_ids = [1, 4]  # 1-9, list of ids or 'all' for all subjects
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

#######################
# END CFG
#######################


def unified_deep_sda():
    # Data loading
    dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle, subject_ids='all')

    # Select subject, target domain
    index_subject = 0
    target_data = dataset_bcic_iv_2a[index_subject]
    target_data.X = np.tile(target_data.X, (8, 1, 1))
    target_data.y = np.tile(target_data.y, 8)

    # All but selected subjects, source domain
    del dataset_bcic_iv_2a[index_subject]
    source_data = data_splitters.concatenate_sets(dataset_bcic_iv_2a)

    # Train model (or models) on source data
    # kinda models, bc the weights/parameters are shared, so training one sets the parameters for the other

    # Split data into train, valid, test
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(target_data, n_folds, 3)
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(source_data, n_folds, 3)


if __name__ == '__main__':
    unified_deep_sda()
