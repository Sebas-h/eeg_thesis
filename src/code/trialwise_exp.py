import logging
import os.path
import time
from collections import OrderedDict
import sys
import pickle
import datetime

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

import braindecode.datautil.splitters as splitters
from braindecode.models.eegnet import EEGNetv4

log = logging.getLogger(__name__)


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda, pickle_path=None):
    max_epochs = 2
    max_increase_epochs = 160
    batch_size = 60
    run_after_early_stop = False
    
    ival = [-500, 4000]
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.2
    
    ### DATA LOADING:

    if pickle_path != None:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        data = data[subject_id - 1]

        # Split data into train, validation and test sets:
        train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)
    else:
        ival = [-500, 4000]
        high_cut_hz = 38
        factor_new = 1e-3
        init_block_size = 1000
        valid_set_fraction = 0.2

        train_filename = 'A{:02d}T.gdf'.format(subject_id)
        test_filename = 'A{:02d}E.gdf'.format(subject_id)
        train_filepath = os.path.join(data_folder, train_filename)
        test_filepath = os.path.join(data_folder, test_filename)
        train_label_filepath = train_filepath.replace('.gdf', '.mat')
        test_label_filepath = test_filepath.replace('.gdf', '.mat')

        train_loader = BCICompetition4Set2A(
            train_filepath, labels_filename=train_label_filepath)
        test_loader = BCICompetition4Set2A(
            test_filepath, labels_filename=test_label_filepath)
        train_cnt = train_loader.load()
        test_cnt = test_loader.load()

        # Preprocessing

        train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                            'EOG-central', 'EOG-right'])
        assert len(train_cnt.ch_names) == 22
        # lets convert to millvolt for numerical stability of next operations
        train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
        train_cnt = mne_apply(
            lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                                filt_order=3,
                                axis=1), train_cnt)
        train_cnt = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                    init_block_size=init_block_size,
                                                    eps=1e-4).T,
            train_cnt)

        test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                        'EOG-central', 'EOG-right'])
        assert len(test_cnt.ch_names) == 22
        test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
        test_cnt = mne_apply(
            lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                                filt_order=3,
                                axis=1), test_cnt)
        test_cnt = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                    init_block_size=init_block_size,
                                                    eps=1e-4).T,
            test_cnt)

        marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                                ('Foot', [3]), ('Tongue', [4])])

        train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
        test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

        train_set, valid_set = split_into_two_sets(
            train_set, first_set_fraction=1-valid_set_fraction)
    
    ### END DATA LOADING


    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length='auto').create_network()
    elif model == 'eegnet':
        model = EEGNetv4(n_chans, n_classes, input_time_length=input_time_length).create_network()


    if cuda:
        model.cuda()
    log.info("Model: \n{:s}".format(str(model)))

    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=batch_size)

    stop_criterion = Or([MaxEpochs(max_epochs), NoDecrease('valid_misclass', max_increase_epochs)])

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=F.nll_loss, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop, cuda=cuda)
    exp.run()
    return exp

def save_exp_to_csv(exp, target_dir_path):
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    file_name = timestamp + ".csv"
    exp.epochs_df.to_csv(target_dir_path + "/" + file_name)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)
    data_folder = "/bla/bla"
    pickle_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
        "/bcic_iv_2a_all_9_subjects.pickle"
    subject_id = 1 # 1-9
    low_cut_hz = 4 # 0 or 4
    model = 'eegnet' #'shallow' or 'deep'
    cuda = th.cuda.is_available()
    exp = run_exp(data_folder, subject_id, low_cut_hz, model, cuda, pickle_path=pickle_folder)
    # Save expirement:
    save_exp_to_csv(exp, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')))
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))