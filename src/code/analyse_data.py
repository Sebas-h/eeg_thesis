import pickle
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def analyse_input_data():
    path_to_data = "/Users/sebas/code/thesis/data/bcic_iv_2a_all_9_subjects.pickle"
    subject_id = 1  # 1-9
    model_type = 'shallow'  # 'shallow' or 'deep'
    cuda = False  # or: torch.cuda.is_available()

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


def analyse_subject_eeg_signals():
    def index_first_occurence(nparray, target):
        for i, val in enumerate(nparray):
            if val == target:
                return i

    from src.pipeline.data_loading import load_bcic_iv_2a_data
    data = load_bcic_iv_2a_data(True, 'all')
    # task 0 (= left hand)
    # task_num = 0
    # channels indices: 7=c3, 9=cz, 11=c4
    channel_idx = 11

    tasks_nums = [0, 1, 2, 3]
    colors = ('b', 'g', 'r', 'c')

    for task_num, color in zip(tasks_nums, colors):
        fig, axs = plt.subplots(nrows=3, ncols=3)
        fig.suptitle(f"Task {task_num + 1}")
        for idx, (ax, subject) in enumerate(zip(axs.flat, data)):
            # task_idx = index_first_occurence(subject.y, task_num)
            # data_to_plot = subject.X[task_idx][channel_idx]
            task_idx = np.where(subject.y == task_num)
            tasks = subject.X[task_idx[0]]
            mean_task = np.mean(tasks, axis=0)

            data_to_plot = mean_task[channel_idx]
            ax.plot(data_to_plot, color)
            ax.set_title(f"Subject {idx + 1}")
            ax.set_ylim([-0.75, 0.75])

        fig.tight_layout()

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def shallow_cnn_800_epochs():
    path = "/Users/sebas/code/thesis/results/AASERIALJOB.47021214.0"

    all_epochs = []
    epoch = []  # epoch_number, train_loss, train_acc, valid_loss, valid_acc

    with open(path, "r") as f:
        for line in f.readlines():
            if "Epoch" in line:
                epoch.append(int(line.split(" ")[-1]))
            elif len(epoch) > 0:
                if "Train" in line and "Loss" in line:
                    epoch.append(float(line.split(' ')[-1]))
                elif "Train" in line and "Acc" in line:
                    epoch.append(float(line.split(' ')[-1].split("%")[0]))
                elif "Valid" in line and "Loss" in line:
                    epoch.append(float(line.split(' ')[-1]))
                elif "Valid" in line and "Acc" in line:
                    epoch.append(float(line.split(' ')[-1].split("%")[0]))
                    all_epochs.append(epoch)
                    epoch = []

    epochs = np.array(all_epochs)

    e1 = 10
    e2 = 11
    a = epochs[e1, -1] / 100
    b = epochs[e2, -1] / 100
    print(f"{e1}:{a}  {e2}:{b} \n\t=> diff = {b - a}")

    plot_res = False

    if plot_res:
        num_epoch_shown = 60
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.set_title('Acc')
        ax0.plot(epochs[:num_epoch_shown, 2], label='Train')
        ax0.plot(epochs[:num_epoch_shown, 4], label='Valid')
        ax0.legend()
        ax1.set_title('Loss')
        ax1.plot(epochs[:num_epoch_shown, 1], label="Train")
        ax1.plot(epochs[:num_epoch_shown, 3], label="Valid")
        ax1.legend()
        fig.tight_layout()
        plt.show()


def analyse_shallow_og_experiment():
    path = '/Users/sebas/code/thesis/results/shallow_cropped/190319_005027.csv'
    df = pd.read_csv(path)
    path = '/Users/sebas/code/thesis/results/230319/results/190323_004855.csv'
    # path = '/Users/sebas/code/thesis/results/190323_165924.csv'
    df2 = pd.read_csv(path)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    start_epoch = 0
    end_epoch = 200

    ax0.set_title('Error Cropped Shallow')
    ax1.set_title('Error Cropped Deep')

    # df.iloc[:, -4] = 1 - df.iloc[:, -4]
    # df.iloc[:, -3] = 1 - df.iloc[:, -3]
    # df.iloc[:, -2] = 1 - df.iloc[:, -2]

    ax0.plot(df.iloc[start_epoch:, -4], label="Train")
    ax0.plot(df.iloc[start_epoch:, -3], label="Valid")
    ax0.plot(df.iloc[start_epoch:, -2], label="Test")
    ax0.set_ylim([-0.1, 1.1])
    ax0.legend()

    ax1.plot(df2.iloc[start_epoch:, -4], label="Train")
    ax1.plot(df2.iloc[start_epoch:, -3], label="Valid")
    ax1.plot(df2.iloc[start_epoch:, -2], label="Test")
    ax1.set_ylim([-0.1, 1.1])
    ax1.legend()

    fig.tight_layout()
    plt.show()


def analyse_shallow_variants():
    paths = [
        '/Users/sebas/code/thesis/results/190319_124018.csv'
        , '/Users/sebas/code/thesis/results/190319_131808.csv'
        , '/Users/sebas/code/thesis/results/190319_134814.csv'
        , '/Users/sebas/code/thesis/results/190319_144135.csv'
        , '/Users/sebas/code/thesis/results/190319_185336.csv'
    ]
    paths = ['/Users/sebas/code/thesis/results/230319/results/190322_173546.csv']
    for path in paths:
        df = pd.read_csv(path)

        fig, ax0 = plt.subplots(nrows=1)

        start_epoch = 0
        # end_epoch = 40

        ax0.set_title(f'Accuracy {path[-17:]}')
        ax0.plot(df.iloc[start_epoch:, 2], label='Train')
        ax0.plot(df.iloc[start_epoch:, 4], label='Valid')
        ax0.set_ylim([0, 1.1])
        ax0.legend()
        fig.tight_layout()

    plt.show()


def analyse_difference_mycsv():
    # path = "/Users/sebas/code/thesis/results/230319/results/190322_173546.csv"
    # path = "/Users/sebas/code/thesis/results/230319/results/190322_183625.csv"
    path = '/Users/sebas/code/thesis/results/shallow_cropped/190319_005027.csv'
    og = True

    df = pd.read_csv(path)
    print("Diffs:")

    if og:
        for index, row in df.iterrows():
            if index == df.shape[0] - 1:
                break
            diff = (1 - df.iloc[index + 1, -3]) - (1 - df.iloc[index, -3])
            print(f"{index} & {index + 1} = {diff}")

    else:
        for index, row in df.iterrows():
            if index == df.shape[0] - 1:
                break
            diff = df.iloc[index + 1, 4] - df.iloc[index, 4]
            print(f"{index} & {index + 1} = {diff}")


if __name__ == "__main__":
    # analyse_difference_mycsv()
    # analyse_shallow_og_experiment()
    # analyse_shallow_variants()
    analyse_subject_eeg_signals()
