import os
import braindecode.datautil.splitters as data_splitters
from src.pipeline import data_loading
from src.pipeline.run_model import RunModel
import numpy as np
from src.unified_deep_sda.dataset import BCICIV2a, SiameseBCICIV2A
import pickle

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
def main(args):
    # Expand arguments
    index_subject = args.subject_index
    index_test_fold = args.test_fold_index
    print("Subject and test fold indices", index_subject, index_test_fold)
    # Run experiment
    unified_deep_sda(index_subject, index_test_fold)


def unified_deep_sda(target_idx, index_test_fold):
    # sfrom_pickle = True
    target_finetune_cls = True

    # Data loading
    target_idx = target_idx
    # if sfrom_pickle:
    #     # from pickle
    #     siamsese_bcic = SiameseBCICIV2A(None, None, 4, 9)
    #     with open('siamese.pickle', 'rb') as f:
    #         a = pickle.load(f)
    #     siamsese_bcic.paired_dataset = a
    # else:
    bcic = BCICIV2a()
    target = bcic.get_subject(target_idx)
    # source = bcic.get_subjects([x for x in range(bcic.n_subjects) if x != target_idx])
    #
    # # Make pairs of target and sources suitable for siamese (two-stream) network:
    # siamsese_bcic = SiameseBCICIV2A(target, source, bcic.n_classes, bcic.n_subjects)
    # siamsese_bcic.create_paired_dataset()
    #
    # train_set, valid_set, test_set = siamsese_bcic.split_into_train_valid_test(n_folds, index_test_fold)
    # run_model = RunModel()
    # path_to_saved_model_dict = run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=target_idx)

    sever_model_state_0_0 = '/home/no316758/projects/eeg_thesis/model_sate_subject_0_0ed87556ffbf45ad90cb02b0871ebfd7.pt'
    
    if target_finetune_cls:
        train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(target, n_folds, index_test_fold)
        run_model = RunModel()
        run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=target_idx,
                     sda_freeze=True,
                     tl_model_state=sever_model_state_0_0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='My args parse experiment')
    parser.add_argument('-s', '--subject-index', type=int, default=0, metavar='N',
                        help='subject index, possible values [0, ..., 8] (default: 0)')
    parser.add_argument('-t', '--test-fold-index', type=int, default=0, metavar='N',
                        help='test fold index, possible values [0, ..., 3] (default: 0)')
    args = parser.parse_args()
    main(args)
