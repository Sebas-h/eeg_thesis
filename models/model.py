from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
import torch
from models.eegnet import EEGNet
from models.deep_convnet import DeepConvNet
from models.shallow_convnet import ShallowConvNet


def get_model(dataset, model_state_dict, config):
    # Set up config
    model_name = config['model']['name']
    is_siamese = (config['experiment']['type'] == 'ccsa_da')
    n_classes = dataset.n_classes
    if is_siamese:
        n_chans = dataset.train_set.X['source'].shape[1]
        input_time_length = dataset.train_set.X['source'].shape[2]
    else:
        n_chans = dataset.train_set.X.shape[1]
        input_time_length = dataset.train_set.X.shape[2]

    # Build model:
    if config['cropped']['use']:
        model = build_cropped_model(model_name, n_chans, n_classes, config)
    else:
        model = build_trialwise_model(model_name, n_chans, n_classes,
                                      input_time_length, is_siamese, config)

    # Potentially load state dict and freeze layers
    if model_state_dict is not None:
        model.load_state_dict(torch.load(model_state_dict))
        if config['experiment']['type'] != 'loo_tl':
            model.freeze_layers()
    return model


def build_trialwise_model(model_name, n_chans, n_classes, input_time_length,
                          is_siamese, config):
    i_feature_alignment_layer = config['model']['i_feature_alignment_layer']

    if model_name == 'eegnet':
        model = EEGNet(n_chans, n_classes,
                       input_time_length=input_time_length,
                       siamese=is_siamese,
                       i_feature_alignment_layer=
                       i_feature_alignment_layer)
    elif model_name == 'deep':
        model = DeepConvNet(n_chans, n_classes,
                            input_time_length=input_time_length,
                            final_conv_length='auto',
                            siamese=is_siamese,
                            i_feature_alignment_layer=
                            i_feature_alignment_layer)
    elif model_name == 'shallow':
        model = ShallowConvNet(n_chans, n_classes,
                               input_time_length=input_time_length,
                               final_conv_length='auto',
                               siamese=is_siamese,
                               i_feature_alignment_layer=
                               i_feature_alignment_layer)
    return model


def build_cropped_model(model_name, n_chans, n_classes, config):
    # input_time_length:
    #   will determine how many crops are processed in parallel
    #   supercrop, number of crops taken through network together
    # final_conv_length:
    #   will determine how many crops are processed in parallel
    #   we manually set the length of the final convolution layer
    #   to some length that makes the receptive field of the
    #   ConvNet smaller than the number of samples in a trial
    cropped_input_time_length = config['cropped']['input_time_length']
    final_conv_length_shallow = config['cropped'][
        'final_conv_length_shallow']
    final_conv_length_deep = config['cropped']['final_conv_length_deep']
    if model_name == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes,
                                input_time_length=cropped_input_time_length,
                                final_conv_length=final_conv_length_shallow) \
            .create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         input_time_length=cropped_input_time_length,
                         final_conv_length=final_conv_length_deep) \
            .create_network()
    to_dense_prediction_model(model)
    return model
