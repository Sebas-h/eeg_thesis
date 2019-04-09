import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

"""
Source:
    https://github.com/locuslab/TCN
"""


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Conv layer 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv layer 2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    import os
    import pickle
    from torch import optim
    from torch.functional import F
    from braindecode.torch_ext.util import np_to_var
    from braindecode.models.deep4 import Deep4Net
    import numpy as np

    pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pickle_path = pickle_dir + "/bcic_iv_2a_all_9_subjects.pickle"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    train_example = data[0]
    inputs = train_example.X[:3]
    targets = train_example.y[0]

    # inputs = np.expand_dims(inputs, axis=0)
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np_to_var(inputs)
    inputs = inputs.permute(0, 3, 2, 1)
    # Model:
    # model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=4)
    # model = MyResNet(MyBasicBlock, num_classes=4)
    # model = Deep4Net(22, 4, 1125, 'auto').create_network()
    model = TemporalConvNet(3, 1)
    print(model)
    optimiser = optim.Adam(model.parameters())

    # Train on one example
    model.train()
    optimiser.zero_grad()
    output = model(inputs)
    print(output)
