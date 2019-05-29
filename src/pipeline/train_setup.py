import numpy as np
import torch as th
import torch.nn.functional as F

import braindecode.experiments.monitors as exp_monitor
from braindecode.models.deep4 import Deep4Net
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.util import np_to_var
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.eegnet import EEGNetv4
from src.pipeline.my_models.resnet import myresnet, resnet18
from src.pipeline.my_models.densenet import densenet121
from src.pipeline.my_models.tcn import TCN
from src.pipeline.my_models.conv_autoencoder import ConvAutoEncoder
from src.unified_deep_sda.siamese_eegnet import SiameseEEGNet
from src.unified_deep_sda.losses import CCSALoss
from braindecode.torch_ext.init import glorot_weight_zero_bias
from src.unified_deep_sda.siamese_deep import SiameseDeep
from src.unified_deep_sda.siamese_shallow import SiameseShallow


class TrainSetup:
    def __init__(self, cropped, train_set, model_name, cuda, batch_size,
                 n_classes, input_time_length, final_conv_length_shallow, final_conv_length_deep, sda_finetune=False):
        # Config:
        self.cropped = cropped
        self.model_name = model_name
        if self.cropped:
            assert self.model_name in ['shallow', 'deep'], "Model not available with cropped training"
        else:
            assert self.model_name in ['shallow', 'deep', 'eegnet', 'myresnet', 'resnet18', 'densenet121', 'tcn',
                                       'eegnet_cae', 'siamese_eegnet', 'siamese_deep', 'siamese_shallow'], \
                "Model not available with trialwise training"

        self.train_set = train_set
        self.batch_size = batch_size
        self.cuda = cuda
        self.n_classes = n_classes
        self.final_conv_length_shallow = final_conv_length_shallow
        self.final_conv_length_deep = final_conv_length_deep

        if self.cropped:
            self.input_time_length = input_time_length
        else:
            self.input_time_length = self.train_set.X.shape[2]
        self.sda_finetune = sda_finetune

        # Set up (setting order necessary):
        self.model = self._set_model()
        self.iterator = self._set_iterator()
        self.loss_function = self._set_loss_function()
        self.compute_pred_labels_func = self._set_compute_pred_labels_func()

    def _set_model(self):
        if self.cropped:
            # input channels to ConvNet, corresponds with EEG channels here
            n_chans = int(self.train_set.X.shape[1])
            # input_time_length: will determine how many crops are processed in parallel
            #   supercrop, number of crops taken through network together
            # final_conv_length: will determine how many crops are processed in parallel
            #   we manually set the length of the final convolution layer to some length
            #   that makes the receptive field of the ConvNet smaller than the number of samples in a trial
            if self.model_name == 'shallow':
                model = ShallowFBCSPNet(n_chans, self.n_classes, input_time_length=self.input_time_length,
                                        final_conv_length=self.final_conv_length_shallow).create_network()
            elif self.model_name == 'deep':
                model = Deep4Net(n_chans, self.n_classes, input_time_length=self.input_time_length,
                                 final_conv_length=self.final_conv_length_deep).create_network()
            to_dense_prediction_model(model)
            return model

        n_chans = int(self.train_set.X.shape[1])  # input channels to ConvNet, corresponds with EEG channels here

        if self.model_name == 'shallow':
            model = ShallowFBCSPNet(n_chans, self.n_classes, input_time_length=self.input_time_length,
                                    final_conv_length='auto').create_network()
        elif self.model_name == 'deep':
            model = Deep4Net(n_chans, self.n_classes, input_time_length=self.input_time_length,
                             final_conv_length='auto').create_network()
        elif self.model_name == 'eegnet':
            model = EEGNetv4(n_chans, self.n_classes, input_time_length=self.input_time_length).create_network()
        elif self.model_name == 'myresnet':
            model = myresnet(num_classes=self.n_classes)
        elif self.model_name == 'resnet18':
            model = resnet18(num_classes=self.n_classes)
        elif self.model_name == 'densenet121':
            model = densenet121(num_classes=self.n_classes)
        elif self.model_name == 'tcn':
            channel_sizes = [25] * 8
            model = TCN(
                input_size=22,
                output_size=self.n_classes,
                num_channels=channel_sizes,
                kernel_size=2,
                dropout=0.2
            )
        elif self.model_name == 'eegnet_cae':
            model = ConvAutoEncoder(in_chans=n_chans, n_classes=self.n_classes)
        elif self.model_name == 'siamese_eegnet':
            if not self.sda_finetune:
                n_chans = int(self.train_set.X.shape[2])
                input_time_length = int(self.train_set.X.shape[3])
                model = SiameseEEGNet(n_chans, self.n_classes, input_time_length=input_time_length)
            else:
                model = SiameseEEGNet(n_chans, self.n_classes, input_time_length=self.input_time_length)

        elif self.model_name == 'siamese_deep':
            if not self.sda_finetune:
                n_chans = int(self.train_set.X.shape[2])
                input_time_length = int(self.train_set.X.shape[3])
                model = SiameseDeep(n_chans, self.n_classes, input_time_length=input_time_length,
                                    final_conv_length='auto')
            else:
                model = SiameseDeep(n_chans, self.n_classes, input_time_length=self.input_time_length,
                                    final_conv_length='auto')

        elif self.model_name == 'siamese_shallow':
            if not self.sda_finetune:
                n_chans = int(self.train_set.X.shape[2])
                input_time_length = int(self.train_set.X.shape[3])
                model = SiameseShallow(n_chans, self.n_classes, input_time_length=input_time_length,
                                       final_conv_length='auto')
            else:
                model = SiameseShallow(n_chans, self.n_classes, input_time_length=self.input_time_length,
                                       final_conv_length='auto')

        return model

    def _set_iterator(self):
        if self.cropped:
            # Determine number of predictions per input/trial, used for cropped batch iterator
            dummy_input = np_to_var(self.train_set.X[:1, :, :, None])
            if self.cuda:
                dummy_input = dummy_input.cuda()
            out = self.model(dummy_input)
            n_preds_per_input = out.cpu().data.numpy().shape[2]
            return CropsFromTrialsIterator(
                batch_size=self.batch_size,
                input_time_length=self.input_time_length,
                n_preds_per_input=n_preds_per_input)
        return BalancedBatchSizeIterator(batch_size=self.batch_size)

    def _set_loss_function(self):
        if self.model_name in ('siamese_eegnet', 'siamese_deep', 'siamese_shallow'):
            return CCSALoss(alpha=0.5)
        if self.model_name == 'eegnet_cae':
            return th.nn.MSELoss()
        if self.cropped:
            return lambda preds, targets: F.nll_loss(th.mean(preds, dim=2, keepdim=False), targets)
        return F.nll_loss

    def _set_compute_pred_labels_func(self):
        if self.model_name == 'eegnet_cae':
            def pred_0(all_preds, dataset):
                return [1]

            return pred_0
        if self.cropped:
            return ComputePredictions(cropped_training=self.cropped,
                                      input_time_length=self.input_time_length).compute_pred_labels
        return ComputePredictions(cropped_training=self.cropped).compute_pred_labels


class ComputePredictions:
    def __init__(self, cropped_training, input_time_length=None):
        self.cropped = cropped_training
        self.input_time_length = input_time_length
        if self.cropped:
            assert input_time_length is not None, "Input time length cannot be None if cropped"

    def compute_pred_labels(self, all_preds, dataset):
        if self.cropped:
            return self._compute_preds_cropped(all_preds, self.input_time_length, dataset)
        else:
            return self._compute_preds_trialwise(all_preds)

    @staticmethod
    def _compute_preds_trialwise(all_preds):
        all_pred_labels = []
        for batch_preds in all_preds:
            pred_labels = np.argmax(batch_preds, axis=1).squeeze()
            all_pred_labels.extend(pred_labels)
        all_pred_labels = np.array(all_pred_labels)
        return all_pred_labels

    @staticmethod
    def _compute_preds_cropped(all_preds, input_time_length, dataset):
        # Assign the predictions to the trials
        preds_per_trial = exp_monitor.compute_preds_per_trial_from_crops(all_preds, input_time_length, dataset.X)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        return predicted_labels
