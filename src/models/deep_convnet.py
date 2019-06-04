import numpy as np
from torch import nn
from torch.nn.functional import elu
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.functions import identity

from src.base.base_model import BaseModel


class DeepConvNet(BaseModel):

    def __init__(self, in_chans,
                 n_classes,
                 input_time_length,
                 final_conv_length,
                 n_filters_time=25,
                 n_filters_spat=25,
                 filter_time_length=10,
                 pool_time_length=3,
                 pool_time_stride=3,
                 n_filters_2=50,
                 filter_length_2=10,
                 n_filters_3=100,
                 filter_length_3=10,
                 n_filters_4=200,
                 filter_length_4=10,
                 first_nonlin=elu,
                 first_pool_mode='max',
                 first_pool_nonlin=identity,
                 later_nonlin=elu,
                 later_pool_mode='max',
                 later_pool_nonlin=identity,
                 drop_prob=0.5,
                 double_time_convs=False,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 stride_before_pool=False,
                 siamese=False,
                 i_feature_alignment_layer=None
                 ):
        super(DeepConvNet, self).__init__()

        if i_feature_alignment_layer is None:
            i_feature_alignment_layer = 4  # default alignment layer
        if final_conv_length == 'auto':
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self

        # conv and pool stride
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride

        # Define kind of pooling used:
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)
        first_pool_class = pool_class[self.first_pool_mode]
        later_pool_class = pool_class[self.later_pool_mode]

        n_filters_conv = self.n_filters_spat

        self.temporal_conv = nn.Sequential(
            Expression(_transpose_time_to_spat),
            nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1),
                      stride=1)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                      (1, self.in_chans), stride=(conv_stride, 1),
                      bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha,
                           affine=True, eps=1e-5),
            Expression(self.first_nonlin),
            first_pool_class(kernel_size=(self.pool_time_length, 1),
                             stride=(pool_stride, 1)),
            Expression(self.first_pool_nonlin)
        )

        self.conv_block2 = self.add_conv_pool_block(n_filters_conv,
                                                    self.n_filters_2,
                                                    self.filter_length_2,
                                                    conv_stride,
                                                    pool_stride,
                                                    later_pool_class)
        self.conv_block3 = self.add_conv_pool_block(self.n_filters_2,
                                                    self.n_filters_3,
                                                    self.filter_length_3,
                                                    conv_stride, pool_stride,
                                                    later_pool_class)
        self.conv_block4 = self.add_conv_pool_block(self.n_filters_3,
                                                    self.n_filters_4,
                                                    self.filter_length_4,
                                                    conv_stride, pool_stride,
                                                    later_pool_class)

        self.eval()
        if self.final_conv_length == 'auto':
            out = np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32))
            out = self.forward_once(out)
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.cls = nn.Sequential(
            nn.Conv2d(self.n_filters_4, self.n_classes,
                      (self.final_conv_length, 1), bias=True),
            nn.LogSoftmax(dim=1),
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

    def add_conv_pool_block(self, n_filters_before, n_filters, filter_length,
                            conv_stride, pool_stride,
                            later_pool_class):
        return nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(n_filters_before, n_filters, (filter_length, 1),
                      stride=(conv_stride, 1),
                      bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters, momentum=self.batch_norm_alpha,
                           affine=True, eps=1e-5),
            Expression(self.later_nonlin),
            later_pool_class(kernel_size=(self.pool_time_length, 1),
                             stride=(pool_stride, 1)),
            Expression(self.later_pool_nonlin)
        )

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


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
