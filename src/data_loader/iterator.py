from braindecode.datautil.iterators import BalancedBatchSizeIterator, \
    CropsFromTrialsIterator
from braindecode.torch_ext.util import np_to_var


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
    return BalancedBatchSizeIterator(batch_size=batch_size)
