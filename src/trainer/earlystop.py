import numpy as np
from copy import deepcopy


def get_stop_criterion(config):
    return EarlyStopping(
        patience=config['train']['early_stop_patience'],
        verbose=False,
        max_epochs=config['train']['max_epochs']
    )


class EarlyStopping:
    """
    Early stops the training if validation loss
    doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, max_epochs=30):
        """
        patience (int): How long to wait after last time validation
        loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation
        loss improvement.
                        Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # my addition:
        self.max_epochs = max_epochs
        self.max_epoch_stop = False
        self.epoch_counter = 0
        self.should_stop = False
        self.checkpoint = None

    def __call__(self, val_loss, model):
        # my addition:
        self.epoch_counter += 1
        if self.epoch_counter >= self.max_epochs:
            self.max_epoch_stop = True

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        # my addition:
        if any([self.max_epoch_stop, self.early_stop]):
            self.should_stop = True

    def save_checkpoint(self, val_loss, model):
        """
        Saves models when validation loss decrease.
        :param val_loss:
        :param model:
        :return:
        """
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        # torch.save(models.state_dict(), 'checkpoint.pt')
        self.checkpoint = deepcopy(model.state_dict())
        self.val_loss_min = val_loss
