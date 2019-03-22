import os, pickle
import braindecode.datautil.splitters as splitters
import transfer_learning.my_model as my_model

# path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
path_to_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
               "/bcic_iv_2a_all_9_subjects.pickle"
subject_id = 1  # 1-9

# Load data:
with open(path_to_data, 'rb') as f:
    data = pickle.load(f)

test_accs = []
for subject_data in data:
    acc = my_model.shallow_convnet(subject_data, num_epochs=40, all_subjects=True)
    test_accs.append(acc)

avg_acc = sum(test_accs) / len(test_accs)
print(f"Average acc of all models across subjects = {avg_acc}")
