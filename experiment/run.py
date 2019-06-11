import logging
import sys
import yaml
import os
import torch
import datetime
import uuid
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint

from util.config import load_cfg
from data_loader.data_loader import get_dataset
from data_loader.iterator import get_iterator
from models.model import get_model
from trainer.setup.losses import get_loss
from trainer.setup.monitors import get_prediction_func
from trainer.setup.earlystop import get_stop_criterion
from trainer.setup.optimizer import get_optmizer
from trainer.trainer import Trainer

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)


def main(args):
    # Load config file
    config = load_cfg(args)

    # Set subject id and valid fold
    subject_id = config['experiment']['subject_id']
    i_valid_fold = config['experiment']['i_valid_fold']
    print(subject_id, i_valid_fold)
    exit()

    # Run experiment
    if config['experiment']['type'] == 'ccsa_da':
        train_siamese_model(subject_id, i_valid_fold, config)
    elif config['experiment']['type'] == 'loo_tl':
        train_model_loo_tl(subject_id, i_valid_fold, config)
    else:
        train_model_once(subject_id, i_valid_fold, config)


def train_siamese_model(subject_id, i_valid_fold, config):
    # Train Siamese model:
    siamese_model_state = train_model_once(subject_id, i_valid_fold,
                                           config)

    # Finetune classifier on target:
    config['experiment']['type'] = 'no_tl'  # Continue without siamese
    train_model_once(subject_id, i_valid_fold, config,
                     model_state_dict=siamese_model_state)


def train_model_loo_tl(subject_id, i_valid_fold, config):
    target_subject_id = subject_id
    dataset_name = config['experiment']['dataset']
    n_subjects = config['data'][dataset_name]['n_subjects']

    source_subject_ids = [i for i in range(1, n_subjects + 1) if
                          i != target_subject_id]

    # Train source model on all subjects except the one chosen
    source_model_state = train_model_once(source_subject_ids, i_valid_fold,
                                          config)
    # Fine tune model on target subject
    train_model_once(target_subject_id, i_valid_fold, config,
                     model_state_dict=source_model_state)


def train_model_once(subject_id, i_valid_fold, config,
                     model_state_dict=None):
    # Data loading
    data = get_dataset(subject_id, i_valid_fold,
                       config['experiment']['dataset'], config)
    # import pickle
    # from base.base_data_loader import BaseDataLoader
    # from braindecode.datautil.splitters import split_into_train_valid_test
    # pickle_path = os.path.abspath(
    #     os.path.join(os.path.dirname(__file__), '..',
    #                  'data/bcic_iv_2a_all_9_subjects.pickle'))
    # with open(pickle_path, 'rb') as f:
    #     data = pickle.load(f)
    # data = data[0]
    # train, valid, test = split_into_train_valid_test(data, 4, 0)
    # data = BaseDataLoader(train, valid, test, 4)

    # Build model architecture
    model = get_model(data, model_state_dict, config)

    # Set iterator and metric function handle
    iterator = get_iterator(model, data, config)
    predict_label_func = get_prediction_func(config)

    # Get function handle of loss
    loss_function = get_loss(config)

    # Build optimizer, learning rate scheduler
    stop_criterion = get_stop_criterion(config)
    optimizer = get_optmizer(model, config)

    print(model)

    # Init trainer and train
    trainer = Trainer(data.train_set, data.validation_set,
                      data.test_set, model, optimizer, iterator,
                      loss_function, stop_criterion,
                      model_constraint=MaxNormDefaultConstraint(),
                      cuda=torch.cuda.is_available(),
                      func_compute_pred_labels=predict_label_func,
                      siamese=(config['experiment']['type'] == 'ccsa_da'))
    trainer.train()

    # Save results
    log_training_results(trainer)
    return save_result_and_model(trainer, model, config)


def log_training_results(trainer):
    # Log training perfomance
    print(trainer.epochs_df)
    # Log test perfomance
    print(trainer.test_result)


def save_result_and_model(trainer, model, config):
    # Generate unique UUID for save files and log it
    unique_id = uuid.uuid4().hex
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Make result directory
    result_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..',
                     f"results/{timestamp}_{config['model']['name']}"
                     f"_{unique_id}"))

    # Check ouput dir exists and possibly create it
    parent_output_dir = os.path.abspath(os.path.join(result_dir, os.pardir))
    assert os.path.exists(parent_output_dir), \
        "Parent directory of given output directory does not exist"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Log result dir:
    print("ResultDir:", result_dir)

    # Save config file:
    with open(f'{result_dir}/config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Save results csv
    f_name = f"{result_dir}/train_results.csv"
    trainer.epochs_df.to_csv(f_name)

    # Save model state (parameters)
    file_name_state_dict = f'{result_dir}/model_sate.pt'
    torch.save(model.state_dict(), file_name_state_dict)
    return file_name_state_dict


def parse_given_arguments():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--subject-id', type=int,
                        metavar='N',
                        help='overrides config subject id')
    parser.add_argument('-i', '--i-fold', type=int,
                        metavar='N',
                        help='overrides config valid fold index (0-based)')
    parser.add_argument('-l', '--i-layer', type=int,
                        metavar='N',
                        help='overrides config '
                             'i_feature_alignment_layer (0-based)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_given_arguments())
