import data_loading
import logging
import numpy as np
from cca.gcca import GCCA

# load data
dataset_bcic_iv_2a = data_loading.load_bcic_iv_2a_data(from_pickle=True, subject_ids='all')

input_data = []
for subject in dataset_bcic_iv_2a:
    input_data.append(subject.X[:, 0, :])
# input_data = np.array(input_data)

# set log level
logging.root.setLevel(level=logging.INFO)

# create data in advance
# a = np.random.rand(50, 50)
# b = np.random.rand(50, 60)
# c = np.random.rand(50, 70)
# d = np.random.rand(50, 80)
# e = np.random.rand(50, 90)
# f = np.random.rand(50, 100)
# g = np.random.rand(50, 110)
# h = np.random.rand(50, 120)
# i = np.random.rand(50, 130)
# j = np.random.rand(50, 140)
# k = np.random.rand(50, 150)

a, b, c, d, e, f, g, h, i = input_data

# create instance of GCCA
gcca = GCCA()

# calculate GCCA
# gcca.fit(a, b, c)
gcca.fit(a, b, c, d, e, f, g, h, i)
# gcca.fit(input_data)

# transform
# gcca.transform(a, b, c)
gcca.transform(a, b, c, d, e, f, g, h, i)
# gcca.transform(input_data)

# save
gcca.save_params("gcca.h5")
# load
# gcca.load_params("gcca.h5")
# plot
# gcca.plot_gcca_result()

print('done')
