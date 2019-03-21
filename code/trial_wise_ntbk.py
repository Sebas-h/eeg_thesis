import logging
from datetime import datetime
import pickle
import os
import sys

import torch.nn.functional as F
from torch import optim
import torch as th
import numpy as np
import pandas as pd

from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.eegnet import EEGNet

from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator, BalancedBatchSizeIterator, ClassBalancedBatchSizeIterator
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
train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)

####################################################################################
####################################################################################

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 4
in_chans = train_set.X.shape[1]

# final_conv_length = auto ensures we only get a single output in the time dimension

model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto').create_network()

# model = Deep4Net(in_chans=in_chans, n_classes=n_classes, 
#                 input_time_length=train_set.X.shape[2], final_conv_length='auto').create_network()

# model = EEGNetv4(in_chans=in_chans, n_classes=n_classes, input_time_length=train_set.X.shape[2]).create_network()

if cuda:
    model.cuda()

print("Model: \n{:s}".format(str(model)))

####################################################################################
####################################################################################

rng = RandomState((2018,8,7))
#optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)

# Need to determine number of batch passes per epoch for cosine annealing
n_epochs = 40
batch_size = 30
# n_updates_per_epoch = len(list(get_balanced_batches(len(train_set.X), rng, shuffle=True, batch_size=30)))

iterator = ClassBalancedBatchSizeIterator(batch_size)
n_updates_per_epoch = len([None for b in iterator.get_batches(train_set, True)])

# Optimzer scheduler
scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)
# schedule_weight_decay must be True for AdamW
optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

####################################################################################
####################################################################################
results_epochs_list = []

for i_epoch in range(n_epochs):
    # i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True, batch_size=30)
    # Set model to training mode
    model.train()
    # for i_trials in i_trials_in_batch:
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=True):
        # Have to add empty fourth dimension to X
        # batch_X = train_set.X[i_trials][:,:,:,None]
        # batch_y = train_set.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters 
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()
    
    res = []

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
        # Here, we will use the entire dataset at once, which is still possible
        # for such smaller datasets. Otherwise we would have to use batches.
        net_in = np_to_var(dataset.X[:,:,:,None])
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(dataset.y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in)
        loss = F.nll_loss(outputs, net_target)
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))

        # save evaluation results of epoch:
        res.append(loss)
        res.append(accuracy)

    # save evaluation results of all epochs
    results_epochs_list.append(res)

####################################################################################
####################################################################################

model.eval()
# Here, we will use the entire dataset at once, which is still possible
# for such smaller datasets. Otherwise we would have to use batches.
net_in = np_to_var(test_set.X[:,:,:,None])
if cuda:
    net_in = net_in.cuda()
net_target = np_to_var(test_set.y)
if cuda:
    net_target = net_target.cuda()
outputs = model(net_in)
loss = F.nll_loss(outputs, net_target)
print("Test Loss: {:.5f}".format(float(var_to_np(loss))))
predicted_labels = np.argmax(var_to_np(outputs), axis=1)
accuracy = np.mean(test_set.y  == predicted_labels)
print("Test Accuracy: {:.1f}%".format(accuracy * 100))

####################################################################################
####################################################################################

# Save results to CSV
df = pd.DataFrame(results_epochs_list, columns=[
                  "train_loss", "train_acc", "valid_loss", "valid_acc"])
df["test_loss"] = np.nan
df["test_acc"] = np.nan
df.iat[-1, -2] = loss
df.iat[-1, -1] = accuracy
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
df.to_csv(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 '..', 'results')) + "/" + timestamp + ".csv"
)