import logging
from datetime import datetime
import pickle
import os
import sys

import torch.nn.functional as F
from torch import optim
import torch as th
import numpy as np

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

path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
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
train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)

def calcstat(labels):
    y = np.bincount(labels)
    ii = np.nonzero(y)[0]
    stat = zip(ii, y[ii])
    print(f"{list(stat)}")

calcstat(train_set.y)
calcstat(valid_set.y)
calcstat(test_set.y)

exit()
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
optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)

# Need to determine number of batch passes per epoch for cosine annealing
n_epochs = 5  # 30
n_updates_per_epoch = len([None for b in iterator.get_batches(train_set, True)])
scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)

# schedule_weight_decay must be True for AdamW
optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

####################################################################################
####################################################################################

# rng = RandomState((2017, 6, 30))

# for i_epoch in range(20):
for i_epoch in range(n_epochs):

    # Set model to training mode
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=True):
        # define the input X and output/target y of the current batch:
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

        # Print the running loss:
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (i_epoch + 1, 1, running_loss))
        running_loss = 0.0

    # Print some statistics each epoch
    # Set model to evaluation mode (so that batch-norm and dropout behave differently than in train mode)
    model.eval()
    print("Epoch {:d}".format(i_epoch))
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
