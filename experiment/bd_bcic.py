import logging
import os
import os.path
import time
from collections import OrderedDict
import sys
import datetime

import numpy as np
import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

from braindecode.models.eegnet import EEGNetv4
from models.eegnet import EEGNet
from models.new_eegnet import NewEEGNet
from data_loader.data_loader import get_dataset
from util.config import load_cfg
from data_loader.process_data.braindecodes_processing import processing_data

log = logging.getLogger(__name__)

"""
BRAINDECODE Example Code
BCICIV2a without cropped (no TL obviously as well)
"""


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda, data):
    ival = [-500, 4000]
    max_epochs = 1600
    max_increase_epochs = 160
    batch_size = 60
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.25

    log.info("==============================")
    log.info("Loading Data...")
    log.info("==============================")
    # processing_data()
    train_set, valid_set, test_set = processing_data(
        data_folder,
        subject_id,
        low_cut_hz,
        high_cut_hz,
        factor_new,
        init_block_size,
        ival,
        valid_set_fraction,
    )
    # train_set = data.train_set
    # valid_set = data.validation_set
    # test_set = data.test_set

    log.info("==============================")
    log.info("Setting Up Model...")
    log.info("==============================")
    set_random_seeds(seed=20190706, cuda=cuda)
    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length = train_set.X.shape[2]
    if model == "shallow":
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    elif model == "deep":
        model = Deep4Net(
            n_chans,
            n_classes,
            input_time_length=input_time_length,
            final_conv_length="auto",
        ).create_network()
    elif model == "eegnet":
        # model = EEGNet(n_chans, n_classes,
        #                input_time_length=input_time_length)
        # model = EEGNetv4(n_chans, n_classes,
        #                  input_time_length=input_time_length).create_network()
        model = NewEEGNet(n_chans, n_classes, input_time_length=input_time_length)

    if cuda:
        model.cuda()

    log.info("==============================")
    log.info("Logging Model Architecture:")
    log.info("==============================")
    log.info("Model: \n{:s}".format(str(model)))

    log.info("==============================")
    log.info("Running Experiment:")
    log.info("==============================")
    optimizer = optim.Adam(model.parameters())

    iterator = BalancedBatchSizeIterator(batch_size=batch_size)

    stop_criterion = Or(
        [MaxEpochs(max_epochs), NoDecrease("valid_misclass", max_increase_epochs)]
    )

    monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    exp = Experiment(
        model,
        train_set,
        valid_set,
        test_set,
        iterator=iterator,
        loss_function=F.nll_loss,
        optimizer=optimizer,
        model_constraint=model_constraint,
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column="valid_misclass",
        run_after_early_stop=True,
        cuda=cuda,
    )
    exp.run()
    return exp


if __name__ == "__main__":
    results_dir_path = None
    if len(sys.argv) > 1:
        results_dir_path = sys.argv[1]

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = "/home/no316758/data/BCICIV_2a_gdf/"
    # data_folder = '/Users/sebas/code/_eeg_data/BCICIV_2a_gdf/'
    low_cut_hz = 4  # 0 or 4
    cuda = True

    # get config
    config = load_cfg(None, cfg_path=f"{results_dir_path}config.yaml")

    # get vars from config
    dataset_name = config["experiment"]["dataset"]
    dataset_subject_count = config["data"][dataset_name]["n_subjects"]
    dataset_path = config["data"][dataset_name]["proc_path"]
    dataset_n_classes = config["data"][dataset_name]["n_classes"]
    experiment_type = config["experiment"]["type"]
    experiment_n_folds = config["experiment"]["n_folds"]
    experiment_i_valid_fold = config["experiment"]["i_valid_fold"]
    model_name = config["model"]["name"]  # 'eegnet' or 'shallow' or 'deep'

    # set subject ids to iterate
    subjects = [x for x in range(1, dataset_subject_count + 1)]

    for subject_id in subjects:
        # get data sets
        data = get_dataset(
            subject_id,
            experiment_i_valid_fold,
            dataset_name,
            dataset_path,
            dataset_n_classes,
            dataset_subject_count,
            experiment_n_folds,
            experiment_type,
        )

        # subject_id = 1  # 1-9
        exp = run_exp(data_folder, subject_id, low_cut_hz, model_name, cuda, data)

        log.info("==============================")
        log.info(f"End Experiment - Last 10 epochs - Subject {subject_id}:")
        log.info("==============================")
        log.info("\n" + str(exp.epochs_df.iloc[-10:]))
        log.info(f"END Subject {subject_id}\n")

        exp.epochs_df.to_csv(
            results_dir_path + f"{subject_id}_{dataset_subject_count}.csv"
        )
