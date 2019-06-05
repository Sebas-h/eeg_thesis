import os
from collections import OrderedDict
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import logging
from braindecode.mne_ext.signalproc import mne_apply
import h5py

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def main():
    # Input and output directories
    data_dir = "/Users/sebas/code/_eeg_data/BCICIV_2a_gdf"
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..',
                     'data/bciciv2a_processed_low_cut_4hz'))

    # Check ouput dir exists and possibly create it
    parent_output_dir = os.path.abspath(os.path.join(output_dir, os.pardir))
    assert os.path.exists(parent_output_dir), \
        "Parent directory of given output directory does not exist"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get file paths:
    train_data_paths, test_data_paths = get_paths_raw_data(data_dir)

    # Frequency filter low cut
    low_cut_hz = 4

    # Process and save data
    save_processed_datatset(train_data_paths, test_data_paths, output_dir,
                            low_cut_hz)


def get_paths_raw_data(data_dir):
    #   subject_ids: list of indices in range [1, ..., 9]
    subject_ids = [x for x in range(1, 10)]

    train_data_paths = [{'gdf': data_dir + f"/A0{subject_id}T.gdf",
                         'mat': data_dir + f"/A0{subject_id}T.mat"}
                        for subject_id in subject_ids]
    test_data_paths = [{'gdf': data_dir + f"/A0{subject_id}E.gdf",
                        'mat': data_dir + f"/A0{subject_id}E.mat"}
                       for subject_id in subject_ids]

    return train_data_paths, test_data_paths


def save_processed_datatset(train_filenames, test_filenames, output_dir,
                            low_cut_hz):
    for train_filename, test_filename in zip(train_filenames, test_filenames):
        subject_id = train_filename['mat'].split('/')[-1][2:3]
        log.info("Processing data...")
        full_train_set = process_bbci_data(train_filename['gdf'],
                                           train_filename['mat'], low_cut_hz)
        test_set = process_bbci_data(test_filename['gdf'],
                                     test_filename['mat'], low_cut_hz)

        log.info("Saving processed data...")
        with h5py.File(f'{output_dir}/{subject_id}_train.h5', 'w') as h5file:
            h5file.create_dataset(f'{subject_id}_train_X',
                                  data=full_train_set.X)
            h5file.create_dataset(f'{subject_id}_train_y',
                                  data=full_train_set.y)
        with h5py.File(f'{output_dir}/{subject_id}_test.h5', 'w') as h5file:
            h5file.create_dataset(f'{subject_id}_test_X', data=test_set.X)
            h5file.create_dataset(f'{subject_id}_test_y', data=test_set.y)

        log.info(f"Done processing data subject {subject_id}\n")


def process_bbci_data(filename, labels_filename, low_cut_hz):
    ival = [-500, 4000]
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000

    loader = BCICompetition4Set2A(filename, labels_filename=labels_filename)
    cnt = loader.load()

    # Preprocessing
    cnt = cnt.drop_channels(
        ['STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    assert len(cnt.ch_names) == 22

    # lets convert to millvolt for numerical stability of next operations
    cnt = mne_apply(lambda a: a * 1e6, cnt)
    cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz,
                               cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), cnt)
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T,
                                                  factor_new=factor_new,
                                                  init_block_size=
                                                  init_block_size,
                                                  eps=1e-4).T, cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


if __name__ == '__main__':
    main()
