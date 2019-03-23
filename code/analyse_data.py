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


def analyse_shallow_OG_experiment():
    path = '/Users/sebas/code/thesis/results/shallow_cropped/190319_005027.csv'
    df = pd.read_csv(path)

    fig, ax1 = plt.subplots(nrows=1)

    start_epoch = 0
    end_epoch = 40

    ax1.set_title('Accuracy 190319_005027.csv')
    
    df.iloc[:, -4] = 1 - df.iloc[:, -4]
    df.iloc[:, -3] = 1 - df.iloc[:, -3]
    df.iloc[:, -2] = 1 - df.iloc[:, -2]

    ax1.plot(df.iloc[start_epoch:end_epoch, -4], label="Train")
    ax1.plot(df.iloc[start_epoch:end_epoch, -3], label="Valid")
    ax1.plot(df.iloc[start_epoch:end_epoch, -2], label="Test")
    ax1.set_ylim([0, 1.1])
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

    for path in paths:
        df = pd.read_csv(path)

        fig, ax0 = plt.subplots(nrows=1)

        start_epoch = 0
        # end_epoch = 40

        ax0.set_title(f'Accuracy {path[-17:]}')
        ax0.plot(df.iloc[start_epoch:,2], label='Train')
        ax0.plot(df.iloc[start_epoch:,4], label='Valid')
        ax0.set_ylim([0, 1.1])
        ax0.legend()

        fig.tight_layout()
        
    plt.show()



if __name__ == "__main__":
    analyse_shallow_OG_experiment()
    # analyse_shallow_variants()
