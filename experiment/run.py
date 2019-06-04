import logging
import sys
import yaml
import os
import torch
import datetime
import uuid
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint

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

    # Run experiment
    if config['model']['siamese']:
        train_siamese_model([subject_id], i_valid_fold, config)
    elif config['experiment']['loo_tl']:
        train_model_loo_tl([subject_id], i_valid_fold, config)
    else:
        train_model_once([subject_id], i_valid_fold, config)


def train_siamese_model(subject_id_list, i_valid_fold, config):
    # Train Siamese model:
    siamese_model_state = train_model_once(subject_id_list, i_valid_fold,
                                           config)

    # Finetune classifier on target:
    config['model']['siamese'] = False  # Continue without siamese
    train_model_once(subject_id_list, i_valid_fold, config,
                     model_state_dict=siamese_model_state)


def train_model_loo_tl(subject_id_list, i_valid_fold, config):
    target_subject_id = subject_id_list
    if config['experiment']['dataset'] == 'bciciv2a':
        n_subjects = 9
    elif config['experiment']['dataset'] == 'hgd':
        n_subjects = 14

    source_subject_ids = [i for i in range(1, n_subjects + 1) if
                          i != target_subject_id[0]]

    # Train source model on all subjects except the one chosen
    source_model_state = train_model_once(source_subject_ids, i_valid_fold,
                                          config)

    # Fine tune model on target subject
    train_model_once(target_subject_id, i_valid_fold, config,
                     model_state_dict=source_model_state)


def train_model_once(subject_id_list, i_valid_fold, config,
                     model_state_dict=None):
    # Data loading
    data = get_dataset(subject_id_list, i_valid_fold, config)

    # Build model architecture
    model = get_model(data, model_state_dict, config)
    print(model)

    # Set iterator and metric function handle
    iterator = get_iterator(model, data, config)
    predict_label_func = get_prediction_func(config)

    # Get function handle of loss
    loss_function = get_loss(config)

    # Build optimizer, learning rate scheduler
    stop_criterion = get_stop_criterion(config)
    optimizer = get_optmizer(model, config)

    # Init trainer and train
    trainer = Trainer(data.train_set, data.validation_set,
                      data.test_set, model, optimizer, iterator,
                      loss_function, stop_criterion,
                      model_constraint=MaxNormDefaultConstraint(),
                      cuda=torch.cuda.is_available(),
                      func_compute_pred_labels=predict_label_func,
                      siamese=config['model']['siamese'])
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


def load_cfg(args):
    """Load a YAML configuration file."""
    yaml_filepath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    with open(yaml_filepath, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    cfg = update_cfg_with_args(cfg, args)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def update_cfg_with_args(cfg, args):
    if args.subject_id is not None:
        cfg['experiment']['subject_id'] = args.subject_id
    if args.i_fold is not None:
        cfg['experiment']['i_valid_fold'] = args.i_fold
    return cfg


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_given_arguments())
