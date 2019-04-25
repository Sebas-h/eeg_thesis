import os
import braindecode.datautil.splitters as data_splitters
from src.pipeline import data_loading
from src.pipeline.run_model import RunModel

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


################################################################################################################
# Data loading
dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle, subject_ids='all')

# Select subject
index_subject = 1
data_subject = dataset_bcic_iv_2a[index_subject]

# All but selected subjects:
del dataset_bcic_iv_2a[index_subject]
data_subjects_allbut1 = data_splitters.concatenate_sets(dataset_bcic_iv_2a)

# Split data into train, valid, test
train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject, n_folds, 0)

# TL with retraining:
# train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
# test_set = None

# TL without retraining:
# train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
# _, _, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)

# Conv AutoEncoder, input and target are the same:
# data_subjects_allbut1.y = data_subjects_allbut1.X
# train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
# test_set = None

################################################################################################################
# FIRST TRAINING ROUND:
subject_id = index_subject + 1
run_model = RunModel()
run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=subject_id)

# SECOND TRAINING ROUND:
# train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 1)
# run_model = RunModel()
# run_model.go(
#     train_set, valid_set, test_set,
#     n_classes=n_classes,
#     subject_id=1,
#     tl_model_state=f'model_sate_s{subject_id}_deep.pt',
#     tl_freeze=False
# )

# TL TRAINING WITH PRETRAINED AUTOENCODER WEIGHTS:
# run_model = RunModel()
# run_model.go(
#     train_set, valid_set, test_set,
#     n_classes=n_classes,
#     subject_id=1,
#     tl_model_state='model_sate_s26_deep.pt',
#     tl_freeze=False,
#     tl_eegnetautoencoder=True
# )
