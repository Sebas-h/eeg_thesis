import logging
from datetime import datetime
import pickle
import os, sys

import torch.nn.functional as F
from torch import optim
import torch as th
import numpy as np
import pandas as pd

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var

from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.datautil.iterators import get_balanced_batches
from numpy.random import RandomState

import braindecode.datautil.splitters as splitters

from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops

####################################################################################
####################################################################################

import os.path
import time, datetime
from collections import OrderedDict
import sys

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

####################################################################################
####################################################################################

# path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
path_to_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
               "/bcic_iv_2a_all_9_subjects.pickle"
subject_id = 1  # 1-9
model_type = 'shallow'  # 'shallow' or 'deep'
cuda = False  # or do: torch.cuda.is_available()

####################################################################################
####################################################################################

# Load data:
with open(path_to_data, 'rb') as f:
    data = pickle.load(f)
data = data[subject_id - 1]

# Split data into train, validation and test sets:
# train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)
# train_valid_set, test_set = splitters.split_into_two_sets(data, first_set_fraction=0.5)
# train_set, valid_set = splitters.split_into_two_sets(train_valid_set, first_set_fraction=0.8)

####################################################################################
####################################################################################


data_folder = '/Users/sebas/code/_eeg_data/BCICIV_2a_gdf/'
subject_id = 1 # 1-9
low_cut_hz = 4 # 0 or 4


ival = [-500, 4000]
input_time_length = 1000
batch_size = 60
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


####################################################################################
####################################################################################

set_random_seeds(seed=20190706, cuda=cuda)  # Set seeds for python random module numpy.random and torch.

# This will determine how many crops are processed in parallel:
input_time_length = 1000
n_classes = 4
n_chans = int(train_set.X.shape[1])  # number of channels

# final_conv_length determines the size of the receptive field of the ConvNet
model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length, final_conv_length=30).create_network()

to_dense_prediction_model(model)

if cuda:
    model.cuda()

print("Model: \n{:s}".format(str(model)))

####################################################################################
####################################################################################

# Because cropped, number of predictions per input/trial has to be determined
dummy_input = np_to_var(train_set.X[:1, :, :, None])  # a single trial, all channels, all measurements
if cuda:
    dummy_input = dummy_input.cuda()
out = model(dummy_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]

print("{:d} predictions per input/trial".format(n_preds_per_input))

# Set size of one training iteration
batch_size = 60
iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,
                                   n_preds_per_input=n_preds_per_input)

####################################################################################
####################################################################################

# optimizer = optim.Adam(model.parameters())
# rng = RandomState((2018, 8, 7))
# optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
# optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
optimizer = optim.Adam(model.parameters())

# Need to determine number of batch passes per epoch for cosine annealing
n_epochs = 40
n_updates_per_epoch = len([None for b in iterator.get_batches(train_set, True)])
scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)

# schedule_weight_decay must be True for AdamW
# optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

model_constraint = MaxNormDefaultConstraint()

####################################################################################
####################################################################################

# rng = RandomState((2017, 6, 30))

results_epochs_list = []

for i_epoch in range(n_epochs):

    # Set model to training mode
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=True):
        # Define the input X and output/target y of the current batch:
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()

        # Remove gradients of last backward pass from all parameters. (zero the parameter gradients)
        optimizer.zero_grad()

        # forward pass, put batch data (net_in) through the network, results in 'outputs'
        outputs = model(net_in)

        # Mean predictions across trial
        # Note that this will give identical gradients to computing
        # a per-prediction loss (at least for the combination of log softmax activation
        # and negative log likelihood loss which we are using here)
        outputs = th.mean(outputs, dim=2, keepdim=False)

        loss = F.nll_loss(outputs, net_target)  # calculate the loss
        loss.backward()                         # calculate gradients (i.e. perform backprop)
        optimizer.step()                        # update parameters based on computed gradients

        model_constraint.apply(model)  # model constraints like done in example 

        # Print the running loss:
        # running_loss += loss.item()
        # print('[%d, %5d] loss: %.3f' % (i_epoch + 1, 1, running_loss))
        # running_loss = 0.0

    # Print some statistics each epoch
    # Set model to evaluation mode (so that batch-norm and dropout behave differently than in train mode)
    model.eval()
    print("Epoch {:d}".format(i_epoch))

    res = []

    for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))

        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        print("{:6s} Loss: {:.5f}".format(setname, loss))

        # Assign the predictions to the trials
        preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                             input_time_length,
                                                             dataset.X)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == dataset.y)
        print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))

        # save evaluation results of epoch:        
        res.append(loss)
        res.append(accuracy)

    # save evaluation results of all epochs
    results_epochs_list.append(res)

####################################################################################
####################################################################################

model.eval()
# Collect all predictions and losses
all_preds = []
all_losses = []
batch_sizes = []

for batch_X, batch_y in iterator.get_batches(test_set, shuffle=False):
    net_in = np_to_var(batch_X)
    if cuda:
        net_in = net_in.cuda()
    net_target = np_to_var(batch_y)
    if cuda:
        net_target = net_target.cuda()
    outputs = model(net_in)
    all_preds.append(var_to_np(outputs))
    outputs = th.mean(outputs, dim=2, keepdim=False)
    loss = F.nll_loss(outputs, net_target)
    loss = float(var_to_np(loss))
    all_losses.append(loss)
    batch_sizes.append(len(batch_X))

# Compute mean per-input loss
loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
               np.mean(batch_sizes))
print("Test Loss: {:.5f}".format(loss))

# Assign the predictions to the trials
preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                     input_time_length,
                                                     test_set.X)
# preds per trial are now trials x classes x timesteps/predictions
# Now mean across timesteps for each trial to get per-trial predictions
meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
accuracy = np.mean(predicted_labels == test_set.y)
print("Test Accuracy: {:.1f}%".format(accuracy * 100))


# Save results to CSV
df = pd.DataFrame(results_epochs_list, columns=["train_loss", "train_acc", "valid_loss", "valid_acc"])
df["test_loss"] = np.nan
df["test_acc"] = np.nan
df.iat[-1,-2]= loss
df.iat[-1,-1]= accuracy
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
df.to_csv(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')) + "/" + timestamp + ".csv"
)
