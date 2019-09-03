import yaml
import os
import logging


def load_cfg(args):
    """Load a YAML configuration file."""
    if args.config:
        cfg = args.config
    else:
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
            if not os.path.exists(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def update_cfg_with_args(cfg, args):
    if args is None:
        return cfg
    if args.subject_id is not None:
        cfg['experiment']['subject_id'] = args.subject_id
    if args.i_fold is not None:
        cfg['experiment']['i_valid_fold'] = args.i_fold
    return cfg


def get_config_item(args):
    """
    checks if config item exists
    takes args like ('parent', 'child', 'item')
        i.e. config['parent']['child']['item']
    """
    raise NotImplementedError
