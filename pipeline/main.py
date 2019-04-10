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
dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle, subject_ids=subject_ids)
data_subject_1 = dataset_bcic_iv_2a[0]
data_subject_4 = dataset_bcic_iv_2a[1]

# Split data into train, valid, test
train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)
# train_set, valid_set = data_splitters.split_into_train_test(data_subject_4, 3, 0)
# test_set = None
################################################################################################################

run_model = RunModel()
run_model.go(train_set, valid_set, test_set, n_classes=n_classes, subject_id=1)

# todo: implement freezing weights/biases (parameters) of model with TL, only train last two layers/blocks for deepnet!
# seond run with tl
# run_model = RunModel()
# train_set, valid_set, test_set = data_splitters.split_into_train_valid_test(data_subject_1, n_folds, 0)
# run_model.go(train_set, valid_set, test_set,
#              n_classes=n_classes,
#              subject_id=1,
#              tl_model_state=f'model_sate_s{4}_deep.pt')
#
# print('done with tl')

################################################################################################################
# train_setup = TrainSetup(
#     cropped=cropped,
#     train_set=train_set,
#     model_name=model_name,
#     cuda=cuda,
#     batch_size=batch_size,
#     n_classes=n_classes,
#     input_time_length=input_time_length,
#     final_conv_length_shallow=final_conv_length_shallow,
#     final_conv_length_deep=final_conv_length_deep
# )
#
# model = train_setup.model
# iterator = train_setup.iterator
# loss_function = train_setup.loss_function
# func_compute_pred_labels = train_setup.compute_pred_labels_func
# stop_criterion = pytorchtools.EarlyStopping(patience=max_increase_epochs, verbose=False, max_epochs=max_epochs)
# model_constraint = MaxNormDefaultConstraint()
# optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
################################################################################################################

################################################################################################################
# # Train the model
# train_model = TrainModel(
#     train_set=train_set,
#     valid_set=valid_set,
#     test_set=test_set,
#     model=model,
#     optimizer=optimizer,
#     iterator=iterator,
#     loss_function=loss_function,
#     stop_criterion=stop_criterion,
#     model_constraint=model_constraint,
#     cuda=cuda,
#     func_compute_pred_labels=func_compute_pred_labels
# )
# train_model.run()
# print('Done')
# print(train_model.epochs_df)
# print(train_model.test_result)
#
# # save results and model
# train_model.epochs_df.to_csv('results.csv')
# th.save(model.state_dict(), f'model_sate_s{subject_ids[0]}.pt')
################################################################################################################


