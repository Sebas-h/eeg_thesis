import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'myresnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x1(in_planes, out_planes, stride=1):
    """3x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride, bias=False)


class MyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 padding=(0, 0),
                 downsample=None,
                 groups=1,
                 norm_layer=None
                 ):
        super(MyBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        # self.conv2 = conv3x1(planes, planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), stride=stride, padding=padding, bias=False)
        self.bn1 = norm_layer(planes)
        # self.elu = nn.ELU(inplace=True)
        self.elu = nn.ELU()

        # self.conv2 = conv3x1(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=padding, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)

        return out


class MyResNet(nn.Module):

    def __init__(self,
                 block,
                 num_classes,
                 layers=None,
                 zero_init_residual=False,
                 width_per_group=64,
                 groups=1,
                 norm_layer=None
                 ):
        super(MyResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        ##################
        # My code
        ##################
        self.inplanes = 48
        # 22, 1125
        # in_chans is among which dimension to take slices of the remaining two dimensions,
        # out channels is number of kernels/filters

        self.dimsuffle = _transpose_time_to_spat
        self.conv_temp = nn.Conv2d(1, 48, kernel_size=(3, 1), padding=(1, 0))
        self.conv_spat = nn.Conv2d(48, 48, kernel_size=(1, 22))
        self.bn1 = nn.BatchNorm2d(48)
        self.elu = nn.ELU()

        self.layer1 = self._make_layer(block, planes=48, blocks=2, stride=1, padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)

        self.layer2 = self._make_layer(block, planes=96, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes=96, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)

        self.layer4 = self._make_layer(block, planes=144, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)
        self.layer5 = self._make_layer(block, planes=144, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)

        self.layer6 = self._make_layer(block, planes=144, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)
        self.layer7 = self._make_layer(block, planes=144, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)

        self.layer8 = self._make_layer(block, planes=144, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)
        self.layer9 = self._make_layer(block, planes=144, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                       norm_layer=norm_layer)

        self.layer10 = self._make_layer(block, planes=144, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                        norm_layer=norm_layer)
        self.layer11 = self._make_layer(block, planes=144, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                        norm_layer=norm_layer)

        self.layer12 = self._make_layer(block, planes=144, blocks=1, stride=(2, 1), padding=(1, 0), groups=groups,
                                        norm_layer=norm_layer)
        self.layer13 = self._make_layer(block, planes=144, blocks=1, stride=1, padding=(1, 0), groups=groups,
                                        norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((10, 1))

        self.final_conv = nn.Conv2d(144, num_classes, (10, 1), bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.squeeze = _squeeze_final_output
        ##################
        # End my code
        ##################

        # Weight initialization stuff (?):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the models by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    # Make res layers, with x num of res blocks in it
    def _make_layer(self, block, planes, blocks, stride=1, padding=(0, 0), groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None

        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, padding, downsample, groups, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # Define forward pass behavior
    def forward(self, x):
        x = self.dimsuffle(x)
        x = self.conv_temp(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(x)

        x = self.layer6(x)
        x = self.layer7(x)

        x = self.layer8(x)
        x = self.layer9(x)

        x = self.layer10(x)
        x = self.layer11(x)

        x = self.layer12(x)
        x = self.layer13(x)

        x = self.avgpool(x)

        x = self.final_conv(x)
        x = self.softmax(x)

        x = self.squeeze(x)
        return x


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


###################################################################################################
###################################################################################################
# OG PyTorsch Resnet implementation below:
###################################################################################################
###################################################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the models by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # added for the EEG data
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, **kwargs)
    # if pretrained:
    #     models.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, **kwargs)
    # if pretrained:
    #     models.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def myresnet(num_classes):
    model = MyResNet(MyBasicBlock, num_classes=num_classes)
    return model


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

    # inputs = np.expand_dims(inputs, axis=0)
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np_to_var(inputs)
    # inputs = inputs.permute(0, 3, 2, 1)

    # Model:
    # models = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=4)
    # models = resnet18(num_classes=4)
    model = MyResNet(MyBasicBlock, num_classes=4)
    # models = Deep4Net(22, 4, 1125, 'auto').create_network()
    optimiser = optim.Adam(model.parameters())

    # Train on one example
    model.train()
    optimiser.zero_grad()
    output = model(inputs)
    print(output)
    # loss = F.nll_loss(output, targets)
    # loss.backward()
    # optimiser.step()
