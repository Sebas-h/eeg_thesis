import numpy as np
from torch import nn
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var

from base.base_model import BaseModel


class ShallowConvNet(BaseModel):

    def __init__(self, in_chans,
                 n_classes,
                 input_time_length=None,
                 n_filters_time=40,
                 filter_time_length=25,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_conv_length=30,
                 conv_nonlin=square,
                 pool_mode='mean',
                 pool_nonlin=safe_log,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5,
                 siamese=False,
                 i_feature_alignment_layer=None
                 ):
        super(ShallowConvNet, self).__init__()

        if i_feature_alignment_layer is None:
            i_feature_alignment_layer = 1  # default alignment layer
        if final_conv_length == 'auto':
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        n_filters_conv = self.n_filters_spat

        self.temporal_conv = nn.Sequential(
            Expression(_transpose_time_to_spat),
            nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1),
                      stride=1)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                      (1, self.in_chans), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha,
                           affine=True),
            Expression(self.conv_nonlin),
            pool_class(kernel_size=(self.pool_time_length, 1),
                       stride=(self.pool_time_stride, 1)),
            Expression(self.pool_nonlin)
        )

        if self.final_conv_length == 'auto':
            out = np_to_var(
                np.ones((1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32))
            out = self.forward_once(out)
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.conv_cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(n_filters_conv, self.n_classes,
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


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
