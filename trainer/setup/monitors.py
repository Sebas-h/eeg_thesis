import numpy as np
import braindecode.experiments.monitors as exp_monitor


def get_prediction_func(config):
    cropped = config['cropped']['use']
    cropped_input_time_length = config['cropped']['input_time_length']
    if cropped:
        return ComputePredictions(
            cropped_training=cropped,
            input_time_length=cropped_input_time_length
        ).compute_pred_labels
    return ComputePredictions(cropped_training=cropped).compute_pred_labels


class ComputePredictions:
    """Create a compute predictions class."""

    def __init__(self, cropped_training, input_time_length=None):
        """
        Initialize compute predictions.

        :param cropped_training:
        :param input_time_length:
        """
        self.cropped = cropped_training
        self.input_time_length = input_time_length
        if self.cropped:
            assert input_time_length is not None,\
                'Input time length cannot be None if cropped'

    def compute_pred_labels(self, all_preds, dataset):
        """
        Compute the predictions per label.

        :param all_preds:
        :param dataset:
        :return:
        """
        if self.cropped:
            return _compute_preds_cropped(
                all_preds, self.input_time_length, dataset,
            )
        return _compute_preds_trialwise(all_preds)


def _compute_preds_trialwise(all_preds):
    all_pred_labels = []
    for batch_preds in all_preds:
        pred_labels = np.argmax(batch_preds, axis=1).squeeze()
        all_pred_labels.extend(pred_labels)
    all_pred_labels = np.array(all_pred_labels)
    return all_pred_labels


def _compute_preds_cropped(all_preds, input_time_length, dataset):
    # Assign the predictions to the trials
    preds_per_trial = exp_monitor.compute_preds_per_trial_from_crops(
        all_preds, input_time_length, dataset.X,
    )
    # preds per trial are now trials x classes x timesteps/predictions
    # Now mean across timesteps for each trial to get per-trial predictions
    meaned_preds_per_trial = np.array(
        [np.mean(prediction, axis=1) for prediction in preds_per_trial],
    )
    return np.argmax(meaned_preds_per_trial, axis=1)
