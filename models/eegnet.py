import numpy as np
import torch as th
from torch import nn
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var

from base.base_model import BaseModel


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
                 drop_prob=0.25,
                 siamese=False,
                 i_feature_alignment_layer=None  # 0-based index modules
                 ):
        super(EEGNet, self).__init__()

        if i_feature_alignment_layer is None:
            i_feature_alignment_layer = 2  # default alignment layer
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
            nn.Conv2d(in_channels=1, out_channels=self.f1,
                      kernel_size=(1, self.kernel_length),
                      stride=1,
                      bias=False,
                      padding=(0, self.kernel_length // 2)),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3)
        )

        self.spatial_conv = nn.Sequential(
            # Spatial conv layer:
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 4), stride=(1, 4))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            # Separable conv layer:
            nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, 16), stride=1,
                      bias=False, groups=self.f1 * self.d,
                      padding=(0, 16 // 2)),
            nn.Conv2d(self.f1 * self.d, self.f2, (1, 1), stride=1, bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = np_to_var(
            np.ones((1, self.in_chans, self.input_time_length, 1),
                    dtype=np.float32))
        out = self.forward_once(out)
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        # Classifier part:
        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.f2, self.n_classes,
                      (n_out_virtual_chans, self.final_conv_length),
                      bias=True),
            nn.LogSoftmax(dim=1),
            # Transpose back to the the logic of _braindecode,
            #   so time in third dimension (axis=2)
            # Transform back to original shape and
            #   squeeze to (batch_size, n_classes) size
            Expression(_transpose_1_0),
            Expression(_squeeze_final_output)
        )

        # Initialize weights of the network
        self.apply(glorot_weight_zero_bias)

        # Set feature space alignment layer, used in siamese training/testing
        assert 0 <= self.i_feature_alignment_layer < len(self._modules), \
            "Given feature space alignment layer does not " \
            "exist for current model"
        self.feature_alignment_layer = \
            list(self._modules.items())[self.i_feature_alignment_layer][0]

    def forward(self, *inputs):
        if self.siamese:
            return self.forward_siamese(*inputs)
        return self.forward_once(*inputs)

    def forward_once(self, x):
        for module in self._modules:
            x = self._modules[module](x)
        return x

    def forward_siamese(self, x):
        target = x['target']
        source = x['source']

        # Compute embeddings:
        for module in self._modules:
            target = self._modules[module](target)
            source = self._modules[module](source)
            if module == self.feature_alignment_layer:
                break
        target_embedding = target
        source_embedding = source

        # Compute cls for modules past the ones used for the embedding:
        start_cls = False
        for module in self._modules:
            if start_cls:
                source = self._modules[module](source)
            if module == self.feature_alignment_layer:
                start_cls = True
        cls = source

        return {'target_embedding': target_embedding,
                'source_embedding': source_embedding,
                'cls': cls}

    def freeze_layers(self):
        for module in self._modules:
            for param in self._modules[module].parameters():
                param.requires_grad = False
            if module == self.feature_alignment_layer:
                break


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
