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


def main(args):
    # Unfold arguments
    index_subject = args.subject_index
    index_test_fold = args.test_fold_index
    print("Subject and test fold indices", index_subject, index_test_fold)

    # Run experiment

    # train_single_subject(index_subject, index_test_fold)
    train_subject_transfer_learning_allbutone(index_subject, index_test_fold)


def train_single_subject(index_subject, index_test_fold):
    # Data loading
    dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle, subject_ids='all')
    # Select subject
    data_subject = dataset_bcic_iv_2a[index_subject]
    # Split data into train, valid, test
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject, n_folds, index_test_fold)
    # TRAINING:
    subject_id = f"{index_subject + 1}_testfldidx_{index_test_fold}"
    run_model = RunModel()
    file_name_state_dict = run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=subject_id)


def train_subject_transfer_learning_allbutone(index_subject, index_test_fold):
    # Data loading
    dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle, subject_ids='all')
    # Select subject
    data_subject = dataset_bcic_iv_2a[index_subject]
    # All but selected subjects:
    del dataset_bcic_iv_2a[index_subject]
    data_subjects_allbut1 = data_splitters.concatenate_sets(dataset_bcic_iv_2a)

    # TL with retraining:
    # train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
    # test_set = None
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subjects_allbut1,
                                                                                n_folds,
                                                                                index_subject)

    # TL without retraining:
    # train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
    # _, _, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)

    # Conv AutoEncoder, input and target are the same:
    # data_subjects_allbut1.y = data_subjects_allbut1.X
    # train_set, valid_set = data_splitters.split_into_train_test(data_subjects_allbut1, 3, 0)
    # test_set = None

    ################################################################################################################
    # FIRST TRAINING ROUND, (pre-)train:
    subject_id = f"{index_subject + 1}-excluded_testfldidx_{index_test_fold}_source_allbutsubject"
    run_model = RunModel()
    file_name_state_dict = run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=subject_id)

    # SECOND TRAINING ROUND, fine-tine/retrain:
    train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject, n_folds, index_subject)
    run_model = RunModel()
    run_model.go(
        train_set, valid_set, test_set,
        n_classes=n_classes,
        subject_id=f"{index_subject + 1}_testfldidx_{index_test_fold}_target",
        tl_model_state=file_name_state_dict,
        tl_freeze=False
    )

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='My args parse experiment')
    parser.add_argument('-s', '--subject-index', type=int, default=0, metavar='N',
                        help='subject index, possible values [0, ..., 8] (default: 0)')
    parser.add_argument('-t', '--test-fold-index', type=int, default=0, metavar='N',
                        help='test fold index, possible values [0, ..., 3] (default: 0)')
    args = parser.parse_args()
    main(args)
