from braindecode.datautil.splitters import split_into_train_test


class BaseDataLoader:
    """
    Base class for all data loaders
    """

    def __init__(self, full_train_set, test_set, i_valid_fold, batch_size,
                 shuffle):
        self.train_set, self.validation_set = self._split_train_valid_sets(
            full_train_set, i_valid_fold)
        self.test_set = test_set

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_idx = 0

    @staticmethod
    def _split_train_valid_sets(full_train_set, i_valid_fold):
        return split_into_train_test(full_train_set, 4, i_valid_fold)

    def get_train_valid_test_sets(self):
        return self.train_set, self.validation_set, self.test_set
