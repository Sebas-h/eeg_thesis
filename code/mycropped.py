import logging
from datetime import datetime
import pickle
import os
import sys

import torch.nn.functional as F
from torch import optim
import torch as th

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
import braindecode.datautil.splitters as splitters

log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'braindecode')))


def run_exp(subject_id, model_type, cuda, path_to_data):
    assert model_type in ['shallow', 'deep']

    # Load data:
    with open(path_to_data, 'rb') as f:
        data = pickle.load(f)
    data = data[subject_id - 1]

    # Split data into train, validation and test sets:
    train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)

    # Set model training parameters
    input_time_length = 1000
    max_epochs = 3  # default = 800
    max_increase_epochs = 80
    batch_size = 60

    # Build model:
    set_random_seeds(seed=20190706, cuda=cuda)  # Set seeds for python random module numpy.random and torch.
    n_classes = 4
    n_chans = int(train_set.X.shape[1])  # number of channels

    # model = sequential pytorch model (conv2D)
    if model_type == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                final_conv_length=30).create_network()
    elif model_type == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length, final_conv_length=2).create_network()

    # Transform a sequential model with strides to a model that outputs dense predictions
    # by removing the strides and instead inserting dilations.
    to_dense_prediction_model(model)

    # Log how the model looks thusfar
    log.info("Model: \n{:s}".format(str(model)))

    # Activate cuda if possible
    if cuda:
        model.cuda()

    # Because cropped, number of predictions per input/trial has to be determined
    dummy_input = np_to_var(train_set.X[:1, :, :, None])  # a single trial, all channels, all measurements
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    # Set optimizer
    optimizer = optim.Adam(model.parameters())

    # Set what one training iteration entails
    iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    # When to stop training
    stop_criterion = Or([MaxEpochs(max_epochs), NoDecrease('valid_misclass', max_increase_epochs)])

    # Keep track of how the model is performing during training:
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(input_time_length=input_time_length), RuntimeMonitor()]

    # 
    model_constraint = MaxNormDefaultConstraint()

    # Set loss function to be minimized
    loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)

    # Initialize and run experiment
    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)

    # bcic_pickle_folder = '/Users/sebas/code/eeg_thesis/eeg_classify/data/pickled_bcic_iv_2a/'
    # bcic_pickle_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/bcic_iv_2a')) + "/"
    path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"

    subject_id = 1  # 1-9
    model_type = 'shallow'  # 'shallow' or 'deep'
    cuda = False

    exp = run_exp(subject_id, model_type, cuda, path_to_data)

    # save result dataframe to csv
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp.epochs_df.to_csv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'my_results')) + "/" + timestamp + ".csv"
    )

    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
