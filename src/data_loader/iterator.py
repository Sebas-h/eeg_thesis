from braindecode.datautil.iterators import BalancedBatchSizeIterator, \
    CropsFromTrialsIterator, get_balanced_batches
from braindecode.torch_ext.util import np_to_var
import numpy as np
from numpy.random import RandomState


def get_iterator(model, dataset, config):
    # Config
    cropped_input_time_length = config['cropped']['input_time_length']
    batch_size = config['train']['batch_size']

    # Set up iterator
    if config['cropped']['use']:
        # Determine number of predictions per input/trial,
        #   used for cropped batch iterator
        dummy_input = np_to_var(dataset.train_set.X[:1, :, :, None])
        if config['cuda']:
            dummy_input = dummy_input.cuda()
        out = model(dummy_input)
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        return CropsFromTrialsIterator(
            batch_size=batch_size,
            input_time_length=cropped_input_time_length,
            n_preds_per_input=n_preds_per_input)
    if config['model']['siamese']:
        return PairedDataBalancedBatchSizeIterator(batch_size=batch_size)
    return BalancedBatchSizeIterator(batch_size=batch_size)


class PairedDataBalancedBatchSizeIterator:
    """
    Create batches of balanced size.

    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    seed: int
        Random seed for initialization of `numpy.RandomState` random generator
        that shuffles the batches.
    """

    def __init__(self, batch_size, seed=328774):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = RandomState(self.seed)

    def get_batches(self, dataset, shuffle):
        n_trials = dataset.y['source'].shape[0]
        batches = get_balanced_batches(n_trials,
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            batch_inds = np.array(batch_inds)
            batch_X = {
                'source': dataset.X['source'][batch_inds],
                'target': dataset.X['target'][batch_inds]
            }
            batch_y = {
                'source': dataset.y['source'][batch_inds],
                'target': dataset.y['target'][batch_inds]
            }

            # add empty fourth dimension if necessary
            if batch_X['source'].ndim == 3:
                batch_X['source'] = batch_X['source'][:, :, :, None]
            if batch_X['target'].ndim == 3:
                batch_X['target'] = batch_X['target'][:, :, :, None]

            yield (batch_X, batch_y)

    def reset_rng(self):
        self.rng = RandomState(self.seed)
