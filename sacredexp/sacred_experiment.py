import logging
import sys
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
    model = 'eegnet'  # 'shallow' or 'deep' or 'eegnet'
    cropped = False  # cropped or trialwise training
    cv = False  # cross validation yes or no
    training = {
        'max_epochs': 800,  # max number of epochs if early stopping criteria not satisfied
        'max_increase_epochs': 90,  # early stopping patience value
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
    subject_ids = 'all'


@ex.automain
def run_exp(model, cropped, training, adamw_optimizer, cropped_params, cv):
    if cropped:
        assert model in ['shallow', 'deep'], "Cropped training only possible with model type 'shallow' or 'deep'"
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
    # cv_folds, test_set = data_loading.get_data(cv=cv)
    subject_datasets = data_loading.get_data(cv=cv)
    # train_set, valid_set, test_set = data_loading.load_data()

    example_input = subject_datasets[0][0][0][0].X

    n_chans = int(example_input.shape[1])  # input channels to ConvNet, corresponds with EEG channels here

    if cropped:
        # This will determine how many crops are processed in parallel
        #   supercrop, number of crops taken through network together
        input_time_length = cropped_params['input_time_length']
        # final_conv_length: will determine how many crops are processed in parallel
        #   we manually set the length of the final convolution layer to some length
        #   that makes the receptive field of the ConvNet smaller than the number of samples in a trial
        if model == 'shallow':
            model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                    final_conv_length=cropped_params['final_conv_length_shallow']).create_network()
        elif model == 'deep':
            model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                             final_conv_length=cropped_params['final_conv_length_deep']).create_network()
        to_dense_prediction_model(model)

        # Determine number of predictions per input/trial, used for cropped batch iterator
        dummy_input = np_to_var(example_input[:1, :, :, None])
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
        input_time_length = example_input.shape[2]
        if model == 'shallow':
            model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                    final_conv_length='auto').create_network()
        elif model == 'deep':
            model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                             final_conv_length='auto').create_network()
        elif model == 'eegnet':
            model = EEGNetv4(n_chans, n_classes, input_time_length=input_time_length).create_network()
        iterator = BalancedBatchSizeIterator(batch_size=batch_size)
        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
        loss_function = F.nll_loss

    # Model training settings:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stop_criterion = Or([MaxEpochs(max_epochs), NoDecrease('valid_misclass', max_increase_epochs)])
    model_constraint = MaxNormDefaultConstraint()

    # Log model
    log.info("Model: \n{:s}".format(str(model)))
    log.info("Optimizer: \n{:s}".format(str(optimizer)))
    ex.info['model'] = str(model)
    ex.info['optimizer'] = str(optimizer)

    if cuda:
        model.cuda()

    for index, subject_dataset in enumerate(subject_datasets):
        train_set, valid_set = subject_dataset[0][0]
        test_set = subject_dataset[1]

    # for fold_index, cv_fold in enumerate(cv_folds):
    #     train_set, valid_set = cv_fold
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
            # final_preds=exp.final_preds,
            # final_targets=exp.final_targets
        )
        ex.info['subject_{}'.format(index)] = info
        # log.info(f"Fold {fold_index + 1} of {len(cv_folds)} done")
        log.info("Last 5 epochs")
        log.info("\n" + str(exp.epochs_df.iloc[-5:]))
