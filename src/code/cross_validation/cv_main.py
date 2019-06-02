import math



"""
process:

create folds using existsing methods

create dict with hyperparameters and values

do cv with a set of hyperparameters and 

"""

# Experiment setup variables
# experiment name:
# [models name]_[cropped or trialwise]_[date]_[time].[csv or smt]

# DATASET
name_dataset = "bcic-iv-2a"
# ival = [-500, 4000]
# high_cut_hz = 38
# factor_new = 1e-3
# init_block_size = 1000
subjects = [1]
low_cut_hz = 4 # 0 or 4

# TRAIN VALID TEST SPLIT RATIO
# train : valid : test = 50% : 25% : 25% 
# valid_set_fraction = 0.2

# MODEL TYPE
model = 'deep' #'shallow' or 'deep' or 'eegnet' or 'resnet' or 'tcn'
# + models details: layers, activations, batch norm, dropout, sense layers, etc.
# Model: 
# Sequential(
#   (dimshuffle): Expression(expression=_transpose_time_to_spat)
#   (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
#   (conv_spat): Conv2d(40, 40, kernel_size=(1, 22), stride=(1, 1), bias=False)
#   (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (conv_nonlin): Expression(expression=square)
#   (pool): AvgPool2d(kernel_size=(75, 1), stride=(1, 1), padding=0)
#   (pool_nonlin): Expression(expression=safe_log)
#   (drop): Dropout(p=0.5)
#   (conv_classifier): Conv2d(40, 4, kernel_size=(30, 1), stride=(1, 1), dilation=(15, 1))
#   (softmax): LogSoftmax()
#   (squeeze): Expression(expression=_squeeze_final_output)
# )

# TRAINING SETUP
input_time_length = 1000
batch_size = 60
cropped = True

# Optimizer and learning rate, weight decay and everything
optimizer = "adam"

# stop criteria:
max_epochs = 800
max_increase_epochs = 80

# LOSS FUNCTION
loss_function = lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)
