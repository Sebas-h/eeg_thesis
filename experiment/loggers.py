# make pretty logger for experiment
# like:
"""
log.info(self.equal_len_title(
        ['Epoch', 'Runtime', 'Train Acc', 'Train Loss', 'Valid Acc',
         'Valid Loss', 'Early Stop']))
log.info(self.equal_len_log(
        [self.epochs_df.shape[0]]
        + list(self.epochs_df.iloc[-1]) +
        [f"{self.stop_criterion.counter}/{self.stop_criterion.patience}"]
    ))
def equal_len_title(self, strings, max_length=12):
    result_string = ''
    for string in strings:
        addition = string + (max_length - len(string)) * ' ' + '| '
        result_string += addition
    return result_string[:-2]

def equal_len_log(self, metrics, max_length=12):
    result_string = ''
    for metric in metrics:
        if type(metric) == int:
            metric = str(metric)
        elif type(metric) == float:
            metric = f"{metric:.5f}"
        addition = metric + (max_length - len(metric)) * ' ' + '| '
        result_string += addition
    return result_string[:-2]
"""
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class Logger(ABC):
    @abstractmethod
    def log_epoch(self, epochs_df):
        raise NotImplementedError("Need to implement the log_epoch function!")