import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from braindecode.models.base import BaseModel
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var, var_to_np

from src.base.base_model import BaseModel


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                     maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(BaseModel):
    """
    EEGNet v4 models from [EEGNet4]_.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(self, in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16,  # usually set to F1*D (?)
                 kernel_length=64,
                 third_kernel_size=(8, 4),
                 drop_prob=0.25
                 ):
        super(EEGNet, self).__init__()

        if final_conv_length == 'auto':
            assert input_time_length is not None

        # Assigns all parameters in init to self.param_name
        self.__dict__.update(locals())
        del self.self

        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        # Convolution accros temporal axis
        self.temporal_conv = nn.Sequential(
            # Rearrange dimensions, dimshuffle,
            #   tranform to shape required by pytorch:
            Expression(_transpose_to_b_1_c_0),
            # Temporal conv layer:
            nn.Conv2d(in_channels=1, out_channels=self.F1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            # Spatial conv layer:
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.F1, padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 4), stride=(1, 4))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            # Separable conv layer:
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1,
                      bias=False, groups=self.F1 * self.D,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = self.sep_conv(
            self.temp_spat_conv(np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32))))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        # Classifier part:
        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.F2, self.n_classes,
                      (n_out_virtual_chans, self.final_conv_length),
                      bias=True),
            nn.LogSoftmax(dim=1),
            # Transpose back to the the logic of braindecode,
            #   so time in third dimension (axis=2)
            # Transform back to original shape and
            #   squeeze to (batch_size, n_classes) size
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output)
        )

        # Initialize weights of the network
        self.apply(glorot_weight_zero_bias)

    def forward(self, x, setname, feature_alignment_layer=1,
                target_finetune_cls=False):
        assert feature_alignment_layer in self.__modules, \
            "Given feature alignment layer does not exist for current models"
        if target_finetune_cls:
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            x = self.cls(x)
            return x
        else:
            # Separate streams '0/1' and add empty dimension at end 'None':
            target = x[:, 0, :, :, None]
            source = x[:, 1, :, :, None]

            for module in self._modules:
                target = module.value(target)
                source = module.value(source)
                if module == feature_alignment_layer:
                    break

            target_embedding = target
            source_embedding = source

            for module in self._modules:
                if module == feature_alignment_layer:
                    cont_cls = True
                if cont_cls:
                    source = module.value(source)

            cls = source

            return {'target_embedding': target_embedding,
                    'source_embedding': source_embedding, 'source_cls': cls}


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
