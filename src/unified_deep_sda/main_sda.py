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
    # index_test_fold = args.test_fold_index
    for index_test_fold in range(8):
        # print("Subject and test fold indices", index_subject, index_test_fold)
        # Run experiment
        unified_deep_sda(index_subject, index_test_fold, fine_tune_cls=False)
        print("End of Subject and test fold indices", index_subject, index_test_fold, "\n")


def unified_deep_sda(target_idx, index_test_fold, fine_tune_cls=False):
    # todo: hacky stuff here:
    src_subject_idx = index_test_fold + 1
    index_test_fold = 3
    print(f"target_subject:{target_idx},source_subject:{src_subject_idx}")

    # Data loading
    target_idx = target_idx
    bcic = BCICIV2a()
    target = bcic.get_subject(target_idx)
    # source = bcic.get_subjects([x for x in range(bcic.n_subjects) if x != target_idx])
    source = [bcic.get_subject(src_subject_idx)]

    # Make pairs of target and sources suitable for siamese (two-stream) network:
    siamsese_bcic = SiameseBCICIV2A(target, source, bcic.n_classes, bcic.n_subjects)
    siamsese_bcic.create_paired_dataset()

    # Split data and train
    train_set, valid_set, test_set = siamsese_bcic.split_into_train_valid_test(n_folds, index_test_fold)
    run_model = RunModel()
    model_state_file = run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=target_idx)

    if fine_tune_cls:
        train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(target, n_folds, index_test_fold)
        run_model = RunModel()
        run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=target_idx,
                     siamese_freeze_layers=True,
                     tl_model_state=model_state_file)


def fine_tune_model(target_idx, index_test_fold):
    server_model_state_path = '/home/no316758/projects/eeg_thesis/model_sate_subject_'
    uuids = [[0, 0, '0ed87556ffbf45ad90cb02b0871ebfd7'], [0, 1, '7779e366b69c4300b5d2cc6196a8aced'],
             [0, 2, 'ac88860fccf64dd1b5d227ddf17fa01d'], [0, 3, '81a44514beb84a749aade3d3d8bc9edc'],
             [1, 0, '8a18d18acc4341488b5806594ce535d8'], [1, 1, '7bbced94370d42caa9141c7d18ee665e'],
             [1, 2, '3ebb85b165ee47638b4d628d3837f8d7'], [1, 3, 'd7e807d13f9b45b1a3881e1b86a53c5e'],
             [2, 0, '47301aaea29140ff832e228a6ed4e087'], [2, 1, '73bdf97d324843029fcad72389b328bd'],
             [2, 2, 'fff43c84bc254b7ba4de38ec25501999'], [2, 3, 'ac008b7f476c4bb1ab4c893f964ad679'],
             [3, 0, 'a96b7efe00e44cf1bd4c4de7fbe07c5c'], [3, 1, 'f414c20bb193497cadc1a9eba52347aa'],
             [3, 2, 'a4d6c474fb41414d9a9c7a9428a6ad2f'], [3, 3, '080b3cf0c3234822ad6f89735796f241'],
             [4, 0, 'd1f17163b97646219179a186037da858'], [4, 1, '35f8646497284bce98b07ed1ad2de4c1'],
             [4, 2, '4c2a2b4ef4544606acc3052c25bb46b8'], [4, 3, 'e98ece3cab404314ad0ee20fa6320855'],
             [5, 0, 'db40e9aa690b4c95b7c45dd179a968a8'], [5, 1, 'cd50e9c692a94e7eae01b2dc39ae2dcb'],
             [5, 2, 'd4ea96ad26344632a3530c30207ee98d'], [5, 3, 'a3dbe29e4db6420a8314b91b27df3f1e'],
             [6, 0, '42a6294b7a524a46a45eca35b2612c32'], [6, 1, '2f64ca4fbd57453194d40b8ac4e51341'],
             [6, 2, '97c79545e1524de192e1f677e715bcde'], [6, 3, 'ee2a36e4f4b94909ab671f2d78654015'],
             [7, 0, '58ca8be2cc1d4c9e9c8656253b334545'], [7, 1, 'c5fab482eee04f6981996e56580e71e7'],
             [7, 2, '1499330658fd47f18c0d2210295c2e9b'], [7, 3, '6a86cbf8bfb748899809367d057c87dc'],
             [8, 0, 'fe81b6cf809f47eeb3806da67a2e50b2'], [8, 1, '747c7d6d3a154f71a51aa13542e8ce82'],
             [8, 2, 'f42be8b6667049eab0d7100aa81f2082'], [8, 3, '603b675fbbb2443dad6351ee6b342066']]

    # path_to_model_state = f"{server_model_state_path}{target_idx}_{uuids[(target_idx * 4) + index_test_fold][-1]}.pt"
    path_to_model_state = '/Users/sebas/code/thesis/src/unified_deep_sda/model_sate_subject_0_29e3ea143d6f4ccab7983d0630d30fac.pt'

    # Data loading
    target_idx = target_idx
    bcic = BCICIV2a()
    target = bcic.get_subject(target_idx)

    # Split data and train
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(target, n_folds, index_test_fold)
    run_model = RunModel()
    run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=target_idx,
                 siamese_eegnet_freeze_conv_layers=True,
                 tl_model_state=path_to_model_state)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='My args parse experiment')
    parser.add_argument('-s', '--subject-index', type=int, default=0, metavar='N',
                        help='subject index, possible values [0, ..., 8] (default: 0)')
    parser.add_argument('-t', '--test-fold-index', type=int, default=3, metavar='N',
                        help='test fold index, possible values [0, ..., 3] (default: 0)')
    args = parser.parse_args()
    main(args)
