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
from models.new_deep import NewDeep4Net
from models.new_shallow import NewShallowNet

from data_loader.data_loader import get_dataset_new
from util.config import load_cfg
from data_loader.process_data.braindecodes_processing import processing_data

import experiment.bd_bcic as bd_bcic
import experiment.loo_tl as loo_tl

log = logging.getLogger(__name__)


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
    # data_folder = "/home/no316758/data/BCICIV_2a_gdf/"
    # data_folder = '/Users/sebas/code/_eeg_data/BCICIV_2a_gdf/'
    # low_cut_hz = 4  # 0 or 4
    cuda = True

    # get config
    config = load_cfg(None, cfg_path=f"{results_dir_path}config.yaml")
    experiment_type = config["experiment"]["type"]

    # get vars from config
    dataset_name = config["experiment"]["dataset"]
    dataset_subject_count = config["data"][dataset_name]["n_subjects"]
    dataset_path = config["data"][dataset_name]["proc_path"]
    dataset_n_classes = config["data"][dataset_name]["n_classes"]
    experiment_type = config["experiment"]["type"]
    experiment_n_folds = config["experiment"]["n_folds"]
    experiment_i_valid_fold = config["experiment"]["i_valid_fold"]
    model_name = config["model"]["name"]  # 'eegnet' or 'shallow' or 'deep'
    experiment_max_epochs = config["train"]["max_epochs"]
    experiment_max_increase_epochs = config["train"]["early_stop_patience"]
    experiment_batch_size = config["train"]["batch_size"]

    if experiment_type == 'no_tl':
        bd_bcic.main(config)
    elif experiment_type == 'loo_tl':
        loo_tl.main(config)
    elif experiment_type == 'ccsa_da':
        pass