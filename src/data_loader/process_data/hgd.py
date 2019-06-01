from collections import OrderedDict
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import logging
import numpy as np
from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
import h5py

log = logging.getLogger(__name__)
log.setLevel('INFO')


def main():
    data_dir = '/home/no316758/data/high-gamma-dataset/data'
    output_dir = '/home/no316758/projects/eeg_thesis/data/' \
                 'hgd_processed_low_cut_4hz'

    # subject_ids: list of indices in range [1, 14]
    subject_ids = [x for x in range(1, 15)]

    train_data_paths = [data_dir + f"/train/{subject_id}.mat" for subject_id in
                        subject_ids]
    test_data_paths = [data_dir + f"/test/{subject_id}.mat" for subject_id in
                       subject_ids]
    low_cut_hz = 4
    debug = False
    save_processed_high_gamma_datatset(train_data_paths, test_data_paths,
                                       output_dir, low_cut_hz, debug)


def save_processed_high_gamma_datatset(train_filenames, test_filenames,
                                       output_dir, low_cut_hz, debug=False):
    for train_filename, test_filename in zip(train_filenames, test_filenames):
        log.info("Processing data...")
        full_train_set = process_bbci_data(train_filename,
                                           low_cut_hz=low_cut_hz,
                                           debug=debug)
        test_set = process_bbci_data(test_filename, low_cut_hz=low_cut_hz,
                                     debug=debug)
        subject_id = train_filename.split('/')[-1].split('.')[0]
        log.info("Saving processed data...")
        with h5py.File(f'{output_dir}/{subject_id}_train.h5', 'w') as h5file:
            h5file.create_dataset(f'{subject_id}_train_X',
                                  data=full_train_set.X)
            h5file.create_dataset(f'{subject_id}_train_y',
                                  data=full_train_set.y)
        with h5py.File(f'{output_dir}/{subject_id}_test.h5', 'w') as h5file:
            h5file.create_dataset(f'{subject_id}_test_X', data=test_set.X)
            h5file.create_dataset(f'{subject_id}_test_y', data=test_set.y)
        log.info(f"Done processing data subject {subject_id}")


def process_bbci_data(filename, low_cut_hz, debug=False):
    """
    As taken from schirrmeister braindecode package
    :param filename: data from subject (train or test)
    :param low_cut_hz: 0 or 4
    :param debug:
    :return: SignalAndTarget
    """
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']

    # We loaded all sensors to always get same cleaning
    #   results independent of sensor selection
    # There is an inbuilt heuristic that tries to use
    #   only EEG channels and that definitely
    #   works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading raw data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                         clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already,
    # since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


if __name__ == '__main__':
    main()
