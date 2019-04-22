import torch as th
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var


class OnlyOne:
    class __OnlyOne:
        def __init__(self, arg):
            self.val = arg

        def __str__(self):
            return repr(self) + self.val

    instance = None

    def __init__(self, arg):
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne(arg)
        else:
            print(th.all(th.eq(arg, OnlyOne.instance.val)))
            # print(arg)
            # print(OnlyOne.instance.val)
            OnlyOne.instance.val = arg

    def __getattr__(self, name):
        return getattr(self.instance, name)


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class PrintToo(nn.Module):
    def forward(self, x):
        OnlyOne(x)
        # print(x)
        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='mean',
                 F1=8,
                 D=2,
                 F2=16,  # usually set to F1*D (?)
                 kernel_length=64,
                 third_kernel_size=(8, 4),
                 drop_prob=0.25):
        super(ConvAutoEncoder, self).__init__()

        self.in_chans = in_chans
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.drop_prob = drop_prob

        self.encoder = nn.Sequential(
            # Print(),

            # dimshuffle:  b(atch) c(hannels) 0 1  --> now to b 1 0 c
            Expression(_transpose_to_b_1_c_0),
            # Print(),

            # conv_temporal
            nn.Conv2d(1, self.F1, (1, self.kernel_length), stride=1, bias=False, padding=(0, self.kernel_length // 2,)),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # Print(),

            # conv_spatial
            Conv2dWithConstraint(
                self.F1, self.F1 * self.D, (self.in_chans, 1), max_norm=1, stride=1, bias=False,
                groups=self.F1,
                padding=(0, 0)),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.drop_prob),
            # Print(),

            # conv_separable_depth
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D,
                      padding=(0, 16 // 2)),
            # Print(),

            # conv_separable_point
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.drop_prob),
            # Print(),
        )

        self.decoder = nn.Sequential(
            # Print(),

            # transpose conv_separable_point and avg pool
            nn.ConvTranspose2d(self.F2, self.F1 * self.D,
                               (1, 8),
                               stride=(1, 8),
                               bias=False,
                               output_padding=(0, 2)
                               ),
            nn.ELU(),
            # Print(),

            # transpose conv_separable_depth
            nn.ConvTranspose2d(self.F1 * self.D, self.F1 * self.D,
                               kernel_size=(1, 16),
                               stride=1,
                               bias=False,
                               groups=self.F1 * self.D,
                               padding=(0, 16 // 2)
                               ),
            nn.ELU(),
            # Print(),

            # Transpose conv with constraint (but without) and avg pool:
            nn.ConvTranspose2d(
                self.F1 * self.D, self.F1,
                kernel_size=(self.in_chans, 4),
                stride=(1, 4),
                bias=False,
                groups=self.F1,
                output_padding=(0, 2)
            ),
            nn.ELU(),
            # Print(),

            # Transpose conv temporal (+ equal sizes)
            nn.ConvTranspose2d(8, 1, kernel_size=(1, 1), stride=1, bias=False),
            nn.ELU(),
            Expression(_equal_dimension),
            # Print(),

            Expression(_shuffle_back),
            # Print(),
        )

    def forward(self, x):
        # print("Encoder module sizes:")
        x = self.encoder(x)
        # print("\nDecoder module sizes:")
        x = self.decoder(x)
        return _squeeze_final_output(x)


def _equal_dimension(x):
    return x[:, :, :, :-1]


def _shuffle_back(x):
    return x.permute(0, 2, 3, 1)


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


if __name__ == '__main__':
    import os
    import pickle
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

    # tens_a = th.Tensor([9, 8, 7, 5])
    # tens_b = th.Tensor([9, 8, 7, 6])
    # print(th.all(th.eq(tens_a, tens_b)))
    # exit()

    # inputs = np.expand_dims(inputs, axis=0)
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np_to_var(inputs)
    # inputs = inputs.permute(0, 3, 2, 1)

    # Model:
    model = ConvAutoEncoder(in_chans=22, n_classes=4)
    optimiser = optim.Adam(model.parameters())

    # Train on one example
    model.train()
    optimiser.zero_grad()
    output = model(inputs)

    loss_crit = nn.MSELoss()
    loss = loss_crit(inputs, output)
    loss.backward()
    optimiser.step()

    print(loss.data.item())
