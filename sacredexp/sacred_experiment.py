import logging
import sys
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F

from braindecode.models.deep4 import Deep4Net
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.eegnet import EEGNetv4
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from experiment import Experiment

# Sacred setup
from sacred import Experiment as SacredExperiment
from sacred.observers import FileStorageObserver
# import the Ingredient and the function we want to use:
import data_loading

ex = SacredExperiment(ingredients=[data_loading.data_ingredient])
# template='/Users/sebas/code/thesis/sacredexp/template.html'
ex.observers.append(FileStorageObserver.create('my_runs'))

# Create logger
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)


@ex.config
def my_config(dataset):
    """ Main config function """
    model_name = 'shallow'  # 'shallow' or 'deep' or 'eegnet'
    cropped = False  # cropped or trialwise training
    cv = False  # cross validation yes or no
    tl_abo = False  # transfer learning, all but one training method
    training = {
        'max_epochs': 900,  # max number of epochs if early stopping criteria not satisfied
        'max_increase_epochs': 100,  # early stopping patience value
        'run_after_early_stop': True,  # see experiment
        'n_classes': dataset['n_classes'],  # num of classes in dataset
        'batch_size': 60,  # number of training examples to train per optimization step
        'cuda': th.cuda.is_available()  # cuda check
    }
    adamw_optimizer = dict(
        lr=1e-3,  # learning rate
        weight_decay=0  # learning rate
    )
    cropped_params = dict(
        input_time_length=1000,
        final_conv_length_shallow=30,
        final_conv_length_deep=2
    )


@ex.named_config
def variant1():
    training = {
        'max_epochs': 3
    }


# Update data Ingredient cfg
@data_loading.data_ingredient.config
def update_cfg():
    # todo: if tl_abo then subject_ids > 1 (prefereably 'all')
    subject_ids = 'all'


@ex.automain
def run_exp(model_name, cropped, training, adamw_optimizer, cropped_params, cv, tl_abo):
    if cropped:
        assert model_name in ['shallow', 'deep'], "Cropped training only possible with model type 'shallow' or 'deep'"
    cuda = training['cuda']
    set_random_seeds(seed=20190706, cuda=cuda)

    # Training and stopping parameters
    n_classes = training['n_classes']
    batch_size = training['batch_size']
    max_epochs = training['max_epochs']
    max_increase_epochs = training['max_increase_epochs']
    run_after_early_stop = training['run_after_early_stop']

    # Optimzer parameters
    lr = adamw_optimizer['lr']
    weight_decay = adamw_optimizer['weight_decay']

    # Load data
    subject_datasets = data_loading.get_data(cv=cv)
    first_train_set = subject_datasets[0][0][0][0]
    n_chans = int(first_train_set.X.shape[1])  # input channels to ConvNet, corresponds with EEG channels here

    if cropped:
        # This will determine how many crops are processed in parallel
        #   supercrop, number of crops taken through network together
        input_time_length = cropped_params['input_time_length']
        # final_conv_length: will determine how many crops are processed in parallel
        #   we manually set the length of the final convolution layer to some length
        #   that makes the receptive field of the ConvNet smaller than the number of samples in a trial
        if model_name == 'shallow':
            model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                    final_conv_length=cropped_params['final_conv_length_shallow']).create_network()
        elif model_name == 'deep':
            model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                             final_conv_length=cropped_params['final_conv_length_deep']).create_network()
        to_dense_prediction_model(model)

        # Determine number of predictions per input/trial, used for cropped batch iterator
        dummy_input = np_to_var(first_train_set.X[:1, :, :, None])
        if cuda:
            dummy_input = dummy_input.cuda()
        out = model(dummy_input)
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        log.info("{:d} predictions per input/trial".format(n_preds_per_input))

        iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,
                                           n_preds_per_input=n_preds_per_input)
        monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                    CroppedTrialMisclassMonitor(input_time_length=input_time_length), RuntimeMonitor()]
        loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)
    else:
        input_time_length = first_train_set.X.shape[2]
        if model_name == 'shallow':
            model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                    final_conv_length='auto').create_network()
        elif model_name == 'deep':
            model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                             final_conv_length='auto').create_network()
        elif model_name == 'eegnet':
            model = EEGNetv4(n_chans, n_classes, input_time_length=input_time_length).create_network()
        iterator = BalancedBatchSizeIterator(batch_size=batch_size)
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        loss_function = F.nll_loss

    # Model training settings:
    stop_criterion = Or([MaxEpochs(max_epochs), NoDecrease('valid_misclass', max_increase_epochs)])
    model_constraint = MaxNormDefaultConstraint()

    # Configure optimizer:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Need to determine number of batch passes per epoch for cosine annealing
    # n_updates_per_epoch = len([None for _ in iterator.get_batches(first_train_set, True)])
    # scheduler = CosineAnnealing((max_epochs + max_increase_epochs) * n_updates_per_epoch)
    # # schedule_weight_decay must be True for AdamW
    # optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

    # Log model
    log.info("Model: \n{:s}".format(str(model)))
    log.info("Optimizer: \n{:s}".format(str(optimizer)))
    ex.info['model'] = str(model)
    ex.info['optimizer'] = str(optimizer.optimizer)

    if cuda:
        model.cuda()

    if tl_abo:
        for i, a in enumerate(data_loading.get_data_tl()):
            train_abo = a[0][0]
            valid_abo = a[0][1]
            train_set = a[1][0]
            valid_set = a[1][1]
            test_set = a[1][2]
            subject = a[2][0]
            subjects = a[2][1]

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # Need to determine number of batch passes per epoch for cosine annealing
            n_updates_per_epoch = len([None for _ in iterator.get_batches(train_abo, True)])
            scheduler = CosineAnnealing((max_epochs + max_increase_epochs) * n_updates_per_epoch)
            # schedule_weight_decay must be True for AdamW
            optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

            exp = Experiment(model, train_abo, valid_abo, test_set=None, iterator=iterator,
                             loss_function=loss_function, optimizer=optimizer,
                             model_constraint=model_constraint,
                             monitors=monitors,
                             stop_criterion=stop_criterion,
                             remember_best_column='valid_misclass',
                             run_after_early_stop=run_after_early_stop, cuda=cuda, ex=ex)
            exp.run()
            # model_state = exp.model.state_dict()
            ex.info['{}_subjects_{}'.format(i, subjects)] = {'epochs_loss_misclass': exp.epochs_df}

            # Reset exp for training on subject
            exp.datasets = OrderedDict((('train', train_set), ('valid', valid_set), ('test', test_set)))
            exp.epochs_df = pd.DataFrame()
            exp.before_stop_df = None
            exp.rememberer = None

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # Need to determine number of batch passes per epoch for cosine annealing
            n_updates_per_epoch = len([None for _ in iterator.get_batches(train_set, True)])
            scheduler = CosineAnnealing((max_epochs + max_increase_epochs) * n_updates_per_epoch)
            # schedule_weight_decay must be True for AdamW
            optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)
            exp.optimizer = optimizer

            exp.run()
            ex.info['{}_subject_{}'.format(i, subject)] = {'epochs_loss_misclass': exp.epochs_df}

            log.info(f"Done with subject {subject}")
            log.info("Last 5 epochs")
            log.info("\n" + str(exp.epochs_df.iloc[-5:]))
    else:
        for index, subject_dataset in enumerate(subject_datasets):
            train_set, valid_set = subject_dataset[0][0]
            test_set = subject_dataset[1]

            exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                             loss_function=loss_function, optimizer=optimizer,
                             model_constraint=model_constraint,
                             monitors=monitors,
                             stop_criterion=stop_criterion,
                             remember_best_column='valid_misclass',
                             run_after_early_stop=run_after_early_stop, cuda=cuda, ex=ex)
            exp.run()
            info = dict(
                epochs_loss_misclass=exp.epochs_df,
            )
            ex.info['subject_{}'.format(index)] = info
            log.info(f"Done with subject {index + 1}")
            log.info("Last 5 epochs")
            log.info("\n" + str(exp.epochs_df.iloc[-5:]))
