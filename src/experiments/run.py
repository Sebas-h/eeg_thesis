import logging
import sys
import yaml
import os
import torch
import datetime
import uuid
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint

from src.data_loader.data_loader import get_dataset
from src.data_loader.iterator import get_iterator
from src.models.model import get_model
from src.trainer.losses import get_loss
from src.trainer.metric import get_prediction_func
from src.trainer.earlystop import get_stop_criterion
from src.trainer.optimizer import get_optmizer
from src.trainer.trainer import Trainer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)


def main():
    # Load config file
    config = load_cfg()

    # Set subject id and valid fold
    subject_id = config['experiment']['subject_id']
    i_valid_fold = config['experiment']['i_valid_fold']

    # Run experiment
    train_single_subject_hgd(subject_id, i_valid_fold, config)


def train_single_subject_hgd(subject_id, i_valid_fold, config):
    # Data loading
    data = get_dataset([subject_id], i_valid_fold, config)

    # Build model architecture
    model = get_model(data, config)

    # Set iterator and metric function handle
    iterator = get_iterator(model, data, config)
    predict_label_func = get_prediction_func(config)

    # Get function handle of loss
    loss_function = get_loss(config)

    # Build optimizer, learning rate scheduler
    stop_criterion = get_stop_criterion(config)
    optimizer = get_optmizer(model, config)

    trainer = Trainer(data.train_set, data.validation_set,
                      data.test_set, model, optimizer, iterator,
                      loss_function, stop_criterion,
                      model_constraint=MaxNormDefaultConstraint(),
                      cuda=torch.cuda.is_available(),
                      func_compute_pred_labels=predict_label_func)
    trainer.train()

    # Log training perfomance
    print(trainer.epochs_df)

    # Log test perfomance
    print(trainer.test_result)

    # Generate unique UUID for save files and log it
    unique_id = uuid.uuid4().hex
    print("UUID:", unique_id)

    # Save results and models
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    f_name = f"{timestamp}_subject_{subject_id}_{unique_id}.csv"
    trainer.epochs_df.to_csv(f_name)

    # Save models state (parameters)
    file_name_state_dict = f'model_sate_subject_{subject_id}_{unique_id}.pt'
    torch.save(model.state_dict(), file_name_state_dict)
    return file_name_state_dict


def load_cfg():
    """Load a YAML configuration file."""
    yaml_filepath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'src/config.yaml'))
    with open(yaml_filepath, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
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


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        default="config.yaml",
                        dest="filename",
                        help="configuration file",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    # args = get_parser().parse_args()
    main()
