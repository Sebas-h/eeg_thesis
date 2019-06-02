class BaseDataLoader:
    """
    Base class for all data loaders
    """

    def __init__(self, train_set, valid_set, test_set, n_classes):
        self.train_set = train_set
        self.validation_set = valid_set
        self.test_set = test_set
        self.n_classes = n_classes

    def get_train_valid_test_sets(self):
        return self.train_set, self.validation_set, self.test_set
