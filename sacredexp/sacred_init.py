from sacred import Experiment
from sacred.observers import FileStorageObserver

import logging
import os
import sys
import torch as th

ex = Experiment()
ex.observers.append(FileStorageObserver.create('my_runs'))


# @ex.config
# def my_configdd():
#     a = 5
#
#
# @ex.automain
# def run_exp(a, x):
#     print(str(x) + ' hello world ' + str(a))
#
# @ex.config
# def my_config():
#     """This is my demo configuration"""
#     logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)
#     data_folder = "/bla/bla"  # path to original raw gdf and mat files from BCIC IV 2a
#     pickle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
#                   "/bcic_iv_2a_all_9_subjects.pickle"  # path to pickle dir
#     subject_id = 1  # 1-9
#     low_cut_hz = 4  # 0 or 4
#     model = 'eegnet'  # 'shallow' or 'deep'
#     cuda = th.cuda.is_available()
#
#     bcic_dataset_cfg = {
#         'ival': [-500, 4000]
#         , 'high_cut_hz': 38
#         , 'factor_new': 1e-3
#         , 'init_block_size': 1000
#         , 'valid_set_fraction': 0.2
#     }
#
#     foo = {
#         'max_epochs': 2,
#         'run_after_early_stop': False
#     }
#
#
# @ex.named_config
# def variant1():
#     foo = {
#         'max_epochs': 3
#     }
