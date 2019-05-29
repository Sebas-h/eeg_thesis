import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.functions import safe_log, square


class SiameseShallow(nn.Module):

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
                 drop_prob=0.5):
        super(SiameseShallow, self).__init__()

        if final_conv_length == 'auto':
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        n_filters_conv = self.n_filters_spat

        self.conv_temp_spat = nn.Sequential(
            Expression(_transpose_time_to_spat),
            nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1), stride=1),
            nn.Conv2d(self.n_filters_time, self.n_filters_spat, (1, self.in_chans), stride=1, bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True),
            Expression(self.conv_nonlin),
            pool_class(kernel_size=(self.pool_time_length, 1), stride=(self.pool_time_stride, 1)),
            Expression(self.pool_nonlin)
        )

        if self.final_conv_length == 'auto':
            out = self.conv_temp_spat(
                np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.conv_cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(n_filters_conv, self.n_classes, (self.final_conv_length, 1), bias=True),
            nn.LogSoftmax(dim=1),
            Expression(_squeeze_final_output)
        )

        # Initialize weights of the network
        self.apply(glorot_weight_zero_bias)

    def forward(self, x, setname, target_finetune_cls=False):
        if target_finetune_cls:
            x = self.conv_temp_spat(x)
            x = self.conv_cls(x)
            return x
        else:
            # Separate streams '0/1' and add empty dimension at end 'None':
            target = x[:, 0, :, :, None]
            source = x[:, 1, :, :, None]

            # Forward pass
            target_embedding = self.conv_temp_spat(target)
            source_embedding = self.conv_temp_spat(source)

            # Only cls on target when on test (i.e. done with training)
            if setname == 'test':
                cls = self.conv_cls(target_embedding)
            else:
                cls = self.conv_cls(source_embedding)

            # Always cls on target set
            # cls = self.cls(source_embedding)

            return {'target_embedding': target_embedding, 'source_embedding': source_embedding, 'source_cls': cls}


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
