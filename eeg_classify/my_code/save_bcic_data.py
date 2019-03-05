import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'braindecode')))
import numpy as np
import matplotlib 

bcic_pickle_folder = '/Users/sebas/code/eeg_thesis/eeg_classify/data/pickled_bcic_iv_2a/'
subject_id = 1  # 1-9
    
# Load data:
with open(f'{bcic_pickle_folder}{subject_id}_train_set.pickle', 'rb') as f:
    train_set = pickle.load(f)
with open(f'{bcic_pickle_folder}{subject_id}_valid_set.pickle', 'rb') as f:
    valid_set = pickle.load(f)
with open(f'{bcic_pickle_folder}{subject_id}_test_set.pickle', 'rb') as f:
    test_set = pickle.load(f)

print('done loading')

print(train_set.X.shape)
print(train_set.y.shape)

print([train_set.X[0]])
print(train_set.y[0])

x = np.concatenate((train_set.X, valid_set.X, test_set.X))
print(x.shape)

y = np.concatenate((train_set.y, valid_set.y, test_set.y))
print(y.shape)

e = np.array([x[:,:0,:0], x[:0, :, :0]])
print(e.shape)
