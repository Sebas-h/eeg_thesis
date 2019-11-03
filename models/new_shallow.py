import numpy as np
from torch import nn
from torch.nn import init

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var


class NewShallowNet(nn.Sequential):
    """
    Shallow ConvNet models from [2]_.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_time_length=None,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length=30,
        conv_nonlin_method=square,
        pool_mode="mean",
        pool_nonlin=safe_log,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
        siamese=False,
        i_feature_alignment_layer=None,
        *args
    ):
        super().__init__(*args)
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self
        self.create_network()

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1, self.n_filters_time, (self.filter_time_length, 1), stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        self.add_module("conv_nonlin", Expression(self.conv_nonlin_method))
        self.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        self.add_module("pool_nonlin", Expression(self.pool_nonlin))
        self.add_module("drop", nn.Dropout(p=self.drop_prob))
        if self.final_conv_length == "auto":
            out = self(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_time_length, 1), dtype=np.float32
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv, self.n_classes, (self.final_conv_length, 1), bias=True
            ),
        )
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

    def forward(self, *inputs):
        if self.siamese:
            return self.forward_siamese(*inputs)
        return self.forward_once(*inputs)

    def forward_once(self, x_input):
        for name, module in self._modules.items():
            x_input = module(x_input)
        return x_input

    def forward_siamese(self, x):
        target = x["target"]
        source = x["source"]

        # Compute embeddings:
        for name, module in self._modules.items():
            target = module(target)
            source = module(source)
            if name == self.feature_alignment_layer:
                break
        target_embedding = target
        source_embedding = source

        # Compute cls for modules past the ones used for the embedding:
        start_cls = False
        for name, module in self._modules.items():
            if start_cls:
                source = module(source)
            elif name == self.feature_alignment_layer:
                start_cls = True
        cls = source

        return {
            "target_embedding": target_embedding,
            "source_embedding": source_embedding,
            "cls": cls,
        }


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

