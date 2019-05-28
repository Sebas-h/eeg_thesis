import numpy as np
import torch as th
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from braindecode.models.base import BaseModel
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.torch_ext.functions import identity


class SiameseDeep(nn.Module):

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
                 stride_before_pool=False):
        super(SiameseDeep, self).__init__()

        if final_conv_length == 'auto':
            assert input_time_length is not None

        # Assigns all parameters in init to self.param_name
        # if any(k in vars(self) for k in vars()):
        #     raise Exception("Var is already present in class. Prevent accidental override")
        # vars(self).update((k, v) for k, v in vars().items() if k != 'self')
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

        self.conv_block1 = nn.Sequential(
            Expression(_transpose_time_to_spat),
            nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1), stride=1),
            nn.Conv2d(self.n_filters_time, self.n_filters_spat, (1, self.in_chans), stride=(conv_stride, 1),
                      bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            Expression(self.first_nonlin),
            first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)),
            Expression(self.first_pool_nonlin)
        )

        self.conv_block2 = self.add_conv_pool_block(n_filters_conv, self.n_filters_2, self.filter_length_2, conv_stride,
                                                    pool_stride, later_pool_class)
        self.conv_block3 = self.add_conv_pool_block(self.n_filters_2, self.n_filters_3, self.filter_length_3,
                                                    conv_stride, pool_stride, later_pool_class)
        self.conv_block4 = self.add_conv_pool_block(self.n_filters_3, self.n_filters_4, self.filter_length_4,
                                                    conv_stride, pool_stride, later_pool_class)

        # self.eval()
        if self.final_conv_length == 'auto':
            out = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(
                np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32))
            ))))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.cls = nn.Sequential(
            nn.Conv2d(self.n_filters_4, self.n_classes, (self.final_conv_length, 1), bias=True),
            nn.LogSoftmax(dim=1),
            Expression(_squeeze_final_output)
        )

        # Initialize weights of the network
        self.apply(glorot_weight_zero_bias)

    def add_conv_pool_block(self, n_filters_before, n_filters, filter_length, conv_stride, pool_stride,
                            later_pool_class):
        return nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(n_filters_before, n_filters, (filter_length, 1), stride=(conv_stride, 1),
                      bias=not self.batch_norm),
            nn.BatchNorm2d(n_filters, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            Expression(self.later_nonlin),
            later_pool_class(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)),
            Expression(self.later_pool_nonlin)
        )

    def forward(self, x, setname, target_finetune_cls=False):
        if target_finetune_cls:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.cls(x)
            return x
        else:
            # Separate streams '0/1' and add empty dimension at end 'None':
            target = x[:, 0, :, :, None]
            source = x[:, 1, :, :, None]

            # Forward pass
            target_embedding = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(target))))
            source_embedding = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(source))))

            # Only cls on target when on test (i.e. done with training)
            if setname == 'test':
                cls = self.cls(target_embedding)
            else:
                cls = self.cls(source_embedding)

            # Always cls on target set
            # cls = self.cls(source_embedding)

            return {'target_embedding': target_embedding, 'source_embedding': source_embedding, 'source_cls': cls}


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


if __name__ == '__main__':
    import os
    import pickle
    import math
    from torch import optim
    from torch.functional import F
    from braindecode.torch_ext.util import np_to_var
    from braindecode.models.deep4 import Deep4Net
    import numpy as np

    pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
    pickle_path = pickle_dir + "/bcic_iv_2a_all_9_subjects.pickle"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    train_example = data[0]
    inputs = train_example.X[:3]
    targets = train_example.y[0]

    # inputs = np.expand_dims(inputs, axis=0)
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np_to_var(inputs)

    # Model:
    model = SiameseDeep(
        in_chans=22,
        n_classes=4,
        input_time_length=1125
    )
    print(model)
    optimiser = optim.Adam(model.parameters())

    # Train on one example
    model.train()
    optimiser.zero_grad()
    output = model(inputs)
    # print(output)
    print(output.shape)
