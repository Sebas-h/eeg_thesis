import os, pickle
import braindecode.datautil.splitters as splitters

# path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
path_to_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
               "/bcic_iv_2a_all_9_subjects.pickle"
subject_id = 1  # 1-9

# Load data:
with open(path_to_data, 'rb') as f:
    data = pickle.load(f)

data_subject = data[subject_id - 1]
del data[subject_id - 1]
data_all_but_one = splitters.concatenate_sets(data)

print(data_subject.X.shape)
print(data_all_but_one.X.shape)

# Split data into train, validation and test sets:
# train_set, valid_set, test_set = splitters.split_into_train_valid_test(data, 4, 0)
