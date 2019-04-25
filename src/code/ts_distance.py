import pickle
import os
from tslearn import metrics
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# pickle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
#               "/bcic_iv_2a_all_9_subjects.pickle"
pickle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + \
              "/bcic_iv_2a_all_9_subjects_lowcut_0.pickle"

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)


# s1 = data[0].X
# s2 = data[1].X

def dist_dtw(s1, s2):
    all_dists = []
    for i in range(s1.shape[0]):
        dist = metrics.dtw(s1[i], s2[i])
        all_dists.append(dist)
    return all_dists


def calc_dist():
    for j in range(9):
        subj1_vs_rest = []
        for i in range(9):
            if i != j:
                subj1_vs_rest.append(dist_dtw(data[j].X, data[i].X))
        sums = []
        for d in subj1_vs_rest:
            sums.append(sum(d))
            print(sum(d))
        print('subject {}, min: {}\n'.format(j + 1, min(sums)))


def show_data(d):
    # labels: [('Left Hand', [1]), ('Right Hand', [2],),('Foot', [3]), ('Tongue', [4])])
    # subject 1, trial 1, channel 1
    # subj = 0
    # trial = 0
    channel = 7  # 7=C3, 9=C4, 11=Cz
    subjects = [0, 1, 2, 3]
    label = 0
    plt_data = []
    for subject in subjects:
        trial_ind = np.where(d[subject].y == label)[0][1]
        plt_data.append(d[subject].X[trial_ind][channel])

    fig, axs = plt.subplots(nrows=len(subjects))

    for ax, d in zip(axs, plt_data):
        ax.set_title("f")
        ax.plot(d)
        # ax.legend()

    fig.tight_layout()
    plt.show()


def average_time_series_per_class(data, subject_id):
    sd = data[subject_id]
    indices = []

    for l in range(np.unique(sd.y).shape[0]):
        inds = np.where(sd.y == l)
        indices.append(inds)

    means = []
    for i_indices in indices:
        x = sd.X[i_indices[0]]
        meanx = np.mean(x, axis=0)
        means.append(meanx)

    return means


def calculate_dist_avg_series():
    channel = 7  # channel (or electrode) of EEG recording to display
    # calculate dists
    for si in range(9):
        m0 = average_time_series_per_class(data, si)
        for i in range(9):
            if i < si or i == si:
                continue
            m = average_time_series_per_class(data, i)
            per_channel_dists = []
            for j in range(len(m)):
                dist = metrics.dtw(m0[j][channel], m[j][channel])
                per_channel_dists.append(dist)
            print(f"subject {si} dist-to {i}: {per_channel_dists}")
        print()


# m1 = average_time_series_per_class(data, 2)
# m2 = average_time_series_per_class(data, 5)
# channel = 7  # channel (or electrode) of EEG recording to display
# Plot average per class time series for chosen channel:
# fig, axs = plt.subplots(nrows=len(m1))
# for i, (ax, d) in enumerate(zip(axs, m1)):
#     ax.set_title("Class/label {}".format(i))
#     ax.plot(d[channel])
#
# for i, (ax, d) in enumerate(zip(axs, m2)):
#     ax.set_title("Class/label {}".format(i))
#     ax.plot(d[channel])
#
# fig.tight_layout()
# plt.show()


def dtw():
    # time series
    x = [1, 2, 3, 4]
    y = [5, 6, 6, 3, 2]

    d_mat = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            d_mat[i][j] = abs(xi - yj)
    print(d_mat)


dtw()
