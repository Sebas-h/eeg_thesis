import os
import data_loading
import braindecode.datautil.splitters as data_splitters
from run_model import RunModel

#######################
# CONFIG AND PARAMETERS`
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
data_subject_1 = dataset_bcic_iv_2a[0]
# data_subject_4 = dataset_bcic_iv_2a[3]
data_subjects_allbut1 = data_splitters.concatenate_sets(dataset_bcic_iv_2a[1:])

# Split data into train, valid, test
# train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)

# TL with retraining:
train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
test_set = None

# TL without retraining:
# train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
# _, _, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)

################################################################################################################

run_model = RunModel()
subject_id = 13
run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=subject_id)

# todo: implement freezing weights/biases (parameters) of model with TL, only train last two layers/blocks for deepnet!
# seond run with tl
run_model = RunModel()
train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)
run_model.go(
    train_set, valid_set, test_set,
    n_classes=n_classes,
    subject_id=1,
    tl_model_state=f'model_sate_s{subject_id}_deep.pt'
)
