from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.eegnet import EEGNetv4
from src.unified_deep_sda.siamese_eegnet import SiameseEEGNet
from src.unified_deep_sda.siamese_deep import SiameseDeep
from src.unified_deep_sda.siamese_shallow import SiameseShallow


def get_model(dataset, config):
    # Set up config
    model_name = config['model']['name']
    cropped_input_time_length = config['cropped']['input_time_length']
    final_conv_length_shallow = config['cropped']['final_conv_length_shallow']
    final_conv_length_deep = config['cropped']['final_conv_length_deep']
    finetune = 0
    # input channels to ConvNet, corresponds with EEG channels here
    n_chans = dataset.train_set.X.shape[1]
    input_time_length = dataset.train_set.X.shape[2]
    n_classes = dataset.n_classes

    # Set up model
    if config['cropped']['use']:
        # input_time_length:
        #   will determine how many crops are processed in parallel
        #   supercrop, number of crops taken through network together
        # final_conv_length:
        #   will determine how many crops are processed in parallel
        #   we manually set the length of the final convolution layer
        #   to some length that makes the receptive field of the
        #   ConvNet smaller than the number of samples in a trial
        if model_name == 'shallow':
            model = ShallowFBCSPNet(n_chans, n_classes,
                                    input_time_length=cropped_input_time_length,
                                    final_conv_length=final_conv_length_shallow).create_network()
        elif model_name == 'deep':
            model = Deep4Net(n_chans, n_classes,
                             input_time_length=cropped_input_time_length,
                             final_conv_length=final_conv_length_deep).create_network()
        to_dense_prediction_model(model)
        return model

    if model_name == 'shallow':
        return ShallowFBCSPNet(n_chans, n_classes,
                               input_time_length=input_time_length,
                               final_conv_length='auto').create_network()
    elif model_name == 'deep':
        return Deep4Net(n_chans, n_classes,
                        input_time_length=input_time_length,
                        final_conv_length='auto').create_network()
    elif model_name == 'eegnet':
        return EEGNetv4(n_chans, n_classes,
                        input_time_length=input_time_length).create_network()

    elif model_name == 'siamese_eegnet':
        if not finetune:
            n_chans = int(dataset.train_set.X.shape[2])
            input_time_length = int(dataset.train_set.X.shape[3])
            return SiameseEEGNet(n_chans, n_classes,
                                 input_time_length=input_time_length)
        else:
            return SiameseEEGNet(n_chans, n_classes,
                                 input_time_length=input_time_length)
    elif model_name == 'siamese_deep':
        if not finetune:
            n_chans = int(dataset.train_set.X.shape[2])
            input_time_length = int(dataset.train_set.X.shape[3])
            return SiameseDeep(n_chans, n_classes,
                               input_time_length=input_time_length,
                               final_conv_length='auto')
        else:
            return SiameseDeep(n_chans, n_classes,
                               input_time_length=input_time_length,
                               final_conv_length='auto')
    elif model_name == 'siamese_shallow':
        if not finetune:
            n_chans = int(dataset.train_set.X.shape[2])
            input_time_length = int(dataset.train_set.X.shape[3])
            return SiameseShallow(n_chans, n_classes,
                                  input_time_length=input_time_length,
                                  final_conv_length='auto')
        else:
            return SiameseShallow(n_chans, n_classes,
                                  input_time_length=input_time_length,
                                  final_conv_length='auto')
