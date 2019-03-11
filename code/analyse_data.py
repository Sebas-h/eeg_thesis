import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
subject_id = 1  # 1-9
model_type = 'shallow'  # 'shallow' or 'deep'
cuda = False  # or do: torch.cuda.is_available()

# Load data:
with open(path_to_data, 'rb') as f:
    datas = pickle.load(f)

for subject_id in range(1, 10):
    data = datas[subject_id - 1]

    labels = data.y

    y = np.bincount(labels)
    ii = np.nonzero(y)[0]
    stat = zip(ii, y[ii])

    print(f"subject {subject_id}: {list(stat)}")
