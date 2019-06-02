import logging
from collections import OrderedDict
from copy import deepcopy
import time

import numpy as np
import pandas as pd
import torch as th

from braindecode.datautil.splitters import concatenate_sets
from braindecode.experiments.loggers import Printer
from braindecode.experiments.stopcriteria import MaxEpochs, ColumnBelow, Or, And
from braindecode.torch_ext.util import np_to_var, set_random_seeds
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor

# import sacred_init
# ex = sacred_init.ex

log = logging.getLogger(__name__)


class RememberBest(object):
    """
    Class to remember and restore
    the parameters of the models and the parameters of the
    optimizer at the epoch with the best performance.

    Parameters
    ----------
    column_name: str
        The lowest value in this column should indicate the epoch with the
        best performance (e.g. misclass might make sense).

    Attributes
    ----------
    best_epoch: int
        Index of best epoch
    """

    def __init__(self, column_name):
        self.column_name = column_name
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self, epochs_df, model, optimizer):
        """
        Remember this epoch: Remember parameter values in case this epoch
        has the best performance so far.

        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
            Dataframe containing the column `column_name` with which performance
            is evaluated.
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))
            log.info("")

    def reset_to_best_model(self, epochs_df, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows
        after best epoch from epochs dataframe.

        Modifies parameters of models and optimizer, changes epochs_df in-place.

        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch + 1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class Experiment(object):
    """
    Class that performs one experiment on training, validation and test set.

    It trains as follows:

    1. Train on training set until a given stop criterion is fulfilled
    2. Reset to the best epoch, i.e. reset parameters of the models and the
       optimizer to the state at the best epoch ("best" according to a given
       criterion)
    3. Continue training on the combined training + validation set until the
       loss on the validation set is as low as it was on the best epoch for the
       training set. (or until the ConvNet was trained twice as many epochs as
       the best epoch to prevent infinite training)

    Parameters
    ----------
    model: `torch.nn.Module`
    train_set: :class:`.SignalAndTarget`
    valid_set: :class:`.SignalAndTarget`
    test_set: :class:`.SignalAndTarget`
    iterator: iterator object
    loss_function: function
        Function mapping predictions and targets to a loss:
        (predictions: `torch.autograd.Variable`,
        targets:`torch.autograd.Variable`)
        -> loss: `torch.autograd.Variable`
    optimizer: `torch.optim.Optimizer`
    model_constraint: object
        Object with apply function that takes models and constraints its
        parameters. `None` for no constraint.
    monitors: list of objects
        List of objects with monitor_epoch and monitor_set method, should
        monitor the traning progress.
    stop_criterion: object
        Object with `should_stop` method, that takes in monitoring dataframe
        and returns if training should stop:
    remember_best_column: str
        Name of column to use for storing parameters of best models. Lowest value
        should indicate best performance in this column.
    run_after_early_stop: bool
        Whether to continue running after early stop
    model_loss_function: function, optional
        Function (models -> loss) to add a models loss like L2 regularization.
        Note that this loss is not accounted for in monitoring at the moment.
    batch_modifier: object, optional
        Object with modify method, that can change the batch, e.g. for data
        augmentation
    cuda: bool, optional
        Whether to use cuda.
    pin_memory: bool, optional
        Whether to pin memory of inputs and targets of batch.
    do_early_stop: bool
        Whether to do an early stop at all. If true, reset to best models
        even in case experiment does not run after early stop.
    reset_after_second_run: bool
        If true, reset to best models when second run did not find a valid loss
        below or equal to the best train loss of first run.
    log_0_epoch: bool
        Whether to compute monitor values and log them before the
        start of training.
    loggers: list of :class:`.Logger`
        How to show computed metrics.

    Attributes
    ----------
    epochs_df: `pandas.DataFrame`
        Monitoring values for all epochs.
    """

    def __init__(self, model, train_set, valid_set, test_set,
                 iterator, loss_function, optimizer, model_constraint,
                 monitors, stop_criterion, remember_best_column,
                 run_after_early_stop,
                 model_loss_function=None,
                 batch_modifier=None, cuda=True, pin_memory=False,
                 do_early_stop=True,
                 reset_after_second_run=False,
                 log_0_epoch=True,
                 loggers=('print',),
                 ex=None):
        if run_after_early_stop or reset_after_second_run:
            assert do_early_stop, "Can only run after early stop or reset after second run if doing an early stop"
        if do_early_stop:
            assert valid_set is not None
            assert remember_best_column is not None
        self.model = model
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', valid_set), ('test', test_set)))
        if valid_set is None:
            self.datasets.pop('valid')
            assert not run_after_early_stop
            assert not do_early_stop
        if test_set is None:
            self.datasets.pop('test')

        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_constraint = model_constraint
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.remember_best_column = remember_best_column
        self.run_after_early_stop = run_after_early_stop
        self.model_loss_function = model_loss_function
        self.batch_modifier = batch_modifier
        self.cuda = cuda
        self.epochs_df = pd.DataFrame()
        self.before_stop_df = None
        self.rememberer = None
        self.pin_memory = pin_memory
        self.do_early_stop = do_early_stop
        self.reset_after_second_run = reset_after_second_run
        self.log_0_epoch = log_0_epoch
        self.loggers = loggers
        self.ex = ex
        self.final_preds = dict()
        self.final_targets = dict()
        self.final_test_df = pd.DataFrame()
        self.final_test_dict = dict()

    def run(self):
        """
        Run complete training.
        """
        self.setup_training()
        log.info("Run until first stop...")
        self.run_until_first_stop()
        if self.do_early_stop:
            # always setup for second stop, in order to get best models
            # even if not running after early stop...
            log.info("Setup for second stop...")
            self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            self.run_until_second_stop()
            if self.reset_after_second_run:
                # if no valid loss was found below the best train loss on 1st
                # run, reset models to the epoch with lowest valid_misclass
                if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach:
                    log.info("Resetting to best epoch {:d}".format(
                        self.rememberer.best_epoch))
                    self.rememberer.reset_to_best_model(self.epochs_df,
                                                        self.model,
                                                        self.optimizer)
        # run last time on test set
        # self.final_test_eval()

    def setup_training(self):
        """
        Setup training, i.e. transform models to cuda,
        initialize monitoring.
        """
        # reset remember best extension in case you rerun some experiment
        if self.do_early_stop:
            self.rememberer = RememberBest(self.remember_best_column)
        if self.loggers == ('print',):
            self.loggers = [Printer()]
        self.epochs_df = pd.DataFrame()
        if self.cuda:
            assert th.cuda.is_available(), "Cuda not available"
            self.model.cuda()

    def run_until_first_stop(self):
        """
        Run training and evaluation using only training set for training
        until stop criterion is fulfilled.
        """
        self.run_until_stop(self.datasets, remember_best=self.do_early_stop)

    def run_until_second_stop(self):
        """
        Run training and evaluation using combined training + validation set
        for training.

        Runs until loss on validation  set decreases below loss on training set
        of best epoch or  until as many epochs trained after as before
        first stop.
        """
        datasets = self.datasets
        datasets['train'] = concatenate_sets([datasets['train'],
                                              datasets['valid']])

        self.run_until_stop(datasets, remember_best=True)

    def run_until_stop(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets until stop criterion is
        fulfilled.

        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters at best epoch.
        """
        if self.log_0_epoch:
            self.monitor_epoch(datasets)
            self.log_epoch()
            if remember_best:
                self.rememberer.remember_epoch(self.epochs_df, self.model,
                                               self.optimizer)

        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets for one epoch.

        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters if this epoch is best epoch.
        """
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        start_train_epoch_time = time.time()
        for inputs, targets in batch_generator:
            if self.batch_modifier is not None:
                inputs, targets = self.batch_modifier.process(inputs, targets)
            # could happen that batch modifier has removed all inputs...
            if len(inputs) > 0:
                self.train_batch(inputs, targets)
        end_train_epoch_time = time.time()
        log.info("Time only for training updates: {:.2f}s".format(end_train_epoch_time - start_train_epoch_time))

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.rememberer.remember_epoch(self.epochs_df, self.model, self.optimizer)

    def train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.

        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`
        """
        self.model.train()
        input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
        target_vars = np_to_var(targets, pin_memory=self.pin_memory)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        if self.model_loss_function is not None:
            loss = loss + self.model_loss_function(self.model)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def eval_on_batch(self, inputs, targets):
        """
        Evaluate given inputs and targets.

        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`

        Returns
        -------
        predictions: `torch.autograd.Variable`
        loss: `torch.autograd.Variable`

        """
        self.model.eval()
        with th.no_grad():
            input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
            target_vars = np_to_var(targets, pin_memory=self.pin_memory)
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            outputs = self.model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            if hasattr(outputs, 'cpu'):
                outputs = outputs.cpu().data.numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().data.numpy() for o in outputs]
            loss = loss.cpu().data.numpy()
        return outputs, loss

    def monitor_epoch(self, datasets):
        """
        Evaluate one epoch for given datasets.

        Stores results in `epochs_df`

        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.

        """
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            all_batch_sizes = []
            all_targets = []
            for batch in self.iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.eval_on_batch(batch[0], batch[1])
                all_preds.append(preds)
                all_losses.append(loss)
                all_batch_sizes.append(len(batch[0]))
                all_targets.append(batch[1])

            for m in self.monitors:
                result_dict = m.monitor_set(setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset)
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)

        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys())
        self.epochs_df = self.epochs_df[list(row_dict.keys())]

    def log_epoch(self):
        """
        Print monitoring values for this epoch.
        """
        for logger in self.loggers:
            logger.log_epoch(self.epochs_df)

    def setup_after_stop_training(self):
        """
        Setup training after first stop.

        Resets parameters to best parameters and updates stop criterion.
        """
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.before_stop_df = deepcopy(self.epochs_df)
        self.rememberer.reset_to_best_model(self.epochs_df, self.model,
                                            self.optimizer)
        loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
        self.stop_criterion = Or(stop_criteria=[
            MaxEpochs(max_epochs=self.rememberer.best_epoch * 2),
            ColumnBelow(column_name='valid_loss', target_value=loss_to_reach)])
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))

    def final_test_eval(self):
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)

        setname = 'test'
        dataset = self.datasets['test']
        all_preds = []
        all_losses = []
        all_batch_sizes = []
        all_targets = []
        for batch in self.iterator.get_batches(dataset, shuffle=False):
            preds, loss = self.eval_on_batch(batch[0], batch[1])
            all_preds.append(preds)
            all_losses.append(loss)
            all_batch_sizes.append(len(batch[0]))
            all_targets.append(batch[1])

        for m in self.monitors:
            result_dict = m.monitor_set(setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset)
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)

        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        row_dict['all_preds'] = all_preds
        row_dict['all_targets'] = all_targets
        row_dict['all_losses'] = all_losses
        row_dict['all_batch_sizes'] = all_batch_sizes
        row_dict['dataset'] = dataset

        self.final_test_dict = row_dict


def my_confusion(all_preds, all_batch_sizes, all_targets):
    threshold_for_binary_case = None
    all_pred_labels = []
    all_target_labels = []
    for i_batch in range(len(all_batch_sizes)):
        preds = all_preds[i_batch]
        # preds could be examples x classes x time
        # or just
        # examples x classes
        # make sure not to remove first dimension if it only has size one
        if preds.ndim > 1:
            only_one_row = preds.shape[0] == 1

            pred_labels = np.argmax(preds, axis=1).squeeze()
            # add first dimension again if needed
            if only_one_row:
                pred_labels = pred_labels[None]
        else:
            assert threshold_for_binary_case is not None, (
                "In case of only one output, please supply the "
                "threshold_for_binary_case parameter")
            # binary classification case... assume logits
            pred_labels = np.int32(preds > threshold_for_binary_case)
        # now examples x time or examples
        all_pred_labels.extend(pred_labels)
        targets = all_targets[i_batch]
        if targets.ndim > pred_labels.ndim:
            # targets may be one-hot-encoded
            targets = np.argmax(targets, axis=1)
        elif targets.ndim < pred_labels.ndim:
            # targets may not have time dimension,
            # in that case just repeat targets on time dimension
            extra_dim = pred_labels.ndim - 1
            targets = np.repeat(np.expand_dims(targets, extra_dim),
                                pred_labels.shape[extra_dim],
                                extra_dim)
        assert targets.shape == pred_labels.shape
        all_target_labels.extend(targets)
    all_pred_labels = np.array(all_pred_labels)
    all_target_labels = np.array(all_target_labels)
    assert all_pred_labels.shape == all_target_labels.shape
    # misclass = 1 - np.mean(all_target_labels == all_pred_labels)
    return all_pred_labels, all_target_labels
