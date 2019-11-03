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
import torch

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

from experiment.bd_bcic import build_exp

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
    experiment_max_epochs = config["train"]["max_epochs"]
    experiment_max_increase_epochs = config["train"]["early_stop_patience"]
    experiment_batch_size = config["train"]["batch_size"]

    # set subject ids to iterate
    subjects = [x for x in range(1, dataset_subject_count + 1)]

    for subject_id in subjects:
        ########################
        # 1st run with SOURCE
        ########################
        # get source data
        source_subject_ids = list(filter(lambda x: x != subject_id, subjects))
        source_data = get_dataset_new(
            source_subject_ids,
            experiment_i_valid_fold,
            dataset_path,
            dataset_n_classes,
            dataset_subject_count,
            experiment_n_folds,
            experiment_type,
        )
        # build experiment to train source model
        source_exp = build_exp(
            model_name,
            cuda,
            source_data,
            experiment_batch_size,
            experiment_max_epochs,
            experiment_max_increase_epochs,
        )
        source_exp.run()

        log.info("==============================")
        log.info(
            f"End Source Experiment - Last 10 epochs - Subject {subject_id}:"
        )
        log.info("==============================")
        log.info("\n" + str(source_exp.epochs_df.iloc[-10:]))
        log.info(f"END Source Subject {subject_id}\n")

        # save source result:
        source_exp.epochs_df.to_csv(
            results_dir_path
            + f"leave_{subject_id}_out_{dataset_subject_count}_src.csv"
        )
        # trained source model:
        trained_source_model = source_exp.model.state_dict()
        # save trained source model:
        torch.save(
            trained_source_model, results_dir_path + "source_model_sate.pt"
        )

        ########################
        # 2nd run with TARGET
        ########################
        # get target data:
        target_data = get_dataset_new(
            subject_id,
            experiment_i_valid_fold,
            dataset_path,
            dataset_n_classes,
            dataset_subject_count,
            experiment_n_folds,
            experiment_type,
        )
        target_exp = build_exp(
            model_name,
            cuda,
            target_data,
            experiment_batch_size,
            experiment_max_epochs,
            experiment_max_increase_epochs,
        )
        # set source model as model's initial state:
        target_exp.model.load_state_dict(
            torch.load(results_dir_path + "source_model_sate.pt")
        )
        # run target domain exp
        target_exp.run()
        
        log.info("==============================")
        log.info(
            f"End Target Experiment - Last 10 epochs - Subject {subject_id}:"
        )
        log.info("==============================")
        log.info("\n" + str(source_exp.epochs_df.iloc[-10:]))
        log.info(f"END Target Subject {subject_id}\n")

        # save source result:
        source_exp.epochs_df.to_csv(
            results_dir_path + f"{subject_id}_{dataset_subject_count}_tgt.csv"
        )
