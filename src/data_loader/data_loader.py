import h5py
from braindecode.datautil.signal_target import SignalAndTarget

from src.base.base_data_loader import BaseDataLoader


class HighGammaProcessed(BaseDataLoader):
    def __init__(self, train_h5_file_path, test_h5_file_path, i_valid_fold,
                 batch_size, shuffle):
        # Load sets from h5 files:
        self.full_train_set = self._load_h5py_data(train_h5_file_path)
        self.test_set = self._load_h5py_data(test_h5_file_path)

        super().__init__(self.full_train_set, self.test_set, i_valid_fold,
                         batch_size, shuffle)

    @staticmethod
    def _load_h5py_data(file_path):
        with h5py.File(file_path, 'r') as h5file:
            keys = sorted(list(h5file.keys()))  # 0 is X, 1 is y
            return SignalAndTarget(h5file[keys[0]][()], h5file[keys[1]][()])
