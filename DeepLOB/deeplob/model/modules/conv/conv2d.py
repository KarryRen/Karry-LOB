# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The `Conv2d` Modules of deep-lob net.
    - Conv2d_3Dim: The `Conv2d` Encoder for 3 dimension features.
    - Conv2d_2Dim: The `Conv2d` Encoder for 2 dimension features.
    - Conv2d_1Dim: The `Conv2d` Encoder for 1 dimension features.

"""

from typing import List
import torch
from torch import nn

from ..batch_norm import get_bn_class
from .diy_conv import MultiplicativeConv2d, FractionalConv2d


class Conv2d_3Dim(nn.Module):
    """ The `Conv2d` Encoder for 3 dimension features, such as:
        - `LOB`, shape=(S, L, D, F)
        - `PAF`, shape=(S, PN, D, F)
        - `LOI`, shape=(S, L, D, F)

    The shape of input feature should be (bs, input_c, S, d1, d2, d3),
        where bs, input_c, S is fixed:
            - bs: batch size
            - input_c: the init channel (could be changed)
            - S: feature length
        and [d1, d2, d3] can be changed, [d1, d2, d3] could be [L, D, F] or [PN, D, F] and so on ...
            - L: Level (0~4 represent `level 1 to level 5`)
            - D: Direction (0-bid, 1-ask)
            - F: Feature (0-price, 1-volume, ...)
            - PN: Price Number (0~PN-1 represent `pn 1 to PN`)

    Attention: the input features should be normed before forwarding !!!
    """

    def __init__(
            self, conv_mid_channels: List[int], dim_list: List[int],
            bn_type: str, bn_momentum: float
    ):
        """ Init of Conv2d_3Dim.

        :param conv_mid_channels: the channels of `Conv2d` layers, MUST have 4 int items, first one means input_c, others are mid channels
        :param dim_list: the list of 3dim, should have 3 int items, be corresponded with the encoding feature
        :param bn_type: the type of batch norm layer
        :param bn_momentum: the momentum of get_bn_class(bn_type)

        """

        super(Conv2d_3Dim, self).__init__()

        # ---- Conv 1. Conv Layer for Dim 3 ---- #
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[0], out_channels=conv_mid_channels[1], kernel_size=(1, dim_list[2]), stride=(1, dim_list[2])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[1], momentum=bn_momentum)
        )

        # ---- Conv 2. Conv Layer for Dim 2 ---- #
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[1], out_channels=conv_mid_channels[2], kernel_size=(1, dim_list[1]), stride=(1, dim_list[1])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[2], momentum=bn_momentum)
        )

        # ---- Conv 3. Conv Layer for Dim 1 ---- #
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[2], out_channels=conv_mid_channels[3], kernel_size=(1, dim_list[0]), stride=(1, dim_list[0])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[3], momentum=bn_momentum)
        )

    def forward(self, features_dim3: torch.Tensor):
        """ Forward computing of Conv2d_3Dim.

        :param features_dim3: the input features with 3 dimension, MUST be a 6D tensor.
            shape=(bs, input_c, S, d1, d2, d3), (MUST after Norm !!!).

        :return: encoding x, shape=(bs, mid_c[3], S, 1)

        """

        # ---- Step 1. Get feature shape and reshape to 4D ---- #
        # get the feature shape
        bs, input_c, S = features_dim3.shape[:3]
        # reshape from (bs, input_c, S, d1, d2, d3) to (bs, input_c, S, d1*d2*d3)
        x = torch.reshape(features_dim3, (bs, input_c, S, -1))

        # ---- Step 2. Conv Encoding ---- #
        # d3 Conv, shape from (bs, input_c, S, d1*d2*d3) to (bs, mid_c[1], S, d1*d2)
        x = self.conv_d3(x)
        # d2 Conv, shape from (bs, mid_c[1], S, d1*d2) to (bs, mid_c[1], S, d1)
        x = self.conv_d2(x)
        # d1 Conv, shape from (bs, mid_c[1], S, d1) to (bs, mid_c[3], S, 1)
        x = self.conv_d1(x)
        return x


class Conv2d_2Dim(nn.Module):
    """ The `Conv2d` Encoder for 2 dimension features, such as
        - `OI`, shape=(S, D, F)
        - `VI`, shape=(S, L F)

    The shape of input feature should be (bs, input_c, S, d1, d2),
        where bs, input_c, S is fixed:
            - bs: batch size
            - input_c: the init channel (could be changed)
            - S: feature length
        and [d1, d2] can be changed, [d1, d2] could be (D, F) or (L, F) and so on ...
            - L: Level (0 ~ 4 represent `level 1 to level 5`)
            - D: Direction (0-bid, 1-ask)
            - F: Feature (0-price, 1-volume, ...)

    Attention: the input features should be normed before forwarding !!!
    """

    def __init__(
            self, conv_mid_channels: List[int], dim_list: List[int],
            bn_type: str, bn_momentum: float
    ):
        """ Init of Conv2d_2Dim.

        :param conv_mid_channels: the channels of `Conv2d` layers, should have 3 int items, first one means input_c, others are mid channels
        :param dim_list: the list of 2dim, should have 2 int items, be corresponded with the encoding feature
        :param bn_type: the type of batch norm layer
        :param bn_momentum: the momentum of get_bn_class(bn_type)

        """

        super(Conv2d_2Dim, self).__init__()

        # ---- Conv 1. Conv Layer for Dim 2 ---- #
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[0], out_channels=conv_mid_channels[1], kernel_size=(1, dim_list[1]), stride=(1, dim_list[1])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[1], momentum=bn_momentum)
        )

        # ---- Conv 2. Conv Layer for Dim 1 ---- #
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[1], out_channels=conv_mid_channels[2], kernel_size=(1, dim_list[0]), stride=(1, dim_list[0])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[2], momentum=bn_momentum)
        )

    def forward(self, features_dim2: torch.Tensor):
        """ Forward computing of Conv2d_2Dim.

        :param features_dim2: the input features with 2 dimension, should be a 5D tensor.
            shape=(bs, input_c, S, d1, d2), (MUST after Norm !!!).

        return: encoding x, shape=(bs, mid_c[2], S, 1)

        """

        # ---- Step 1. Get feature shape and reshape to 4D ---- #
        # get the feature shape
        bs, input_c, S = features_dim2.shape[:3]
        # reshape from (bs, input_c, S, d1, d2) to (bs, input_c, S, d1*d2)
        x = torch.reshape(features_dim2, (bs, input_c, S, -1))

        # ---- Step 2. Conv Encoding ---- #
        # d2 Conv, shape from (bs, mid_c[1], S, d1*d2) to (bs, mid_c[1], S, d1)
        x = self.conv_d2(x)
        # d1 Conv, shape from (bs, mid_c[1], S, d1) to (bs, mid_c[3], S, 1)
        x = self.conv_d1(x)
        return x


class Conv2d_1Dim(nn.Module):
    """ The `Conv2d` Encoder for 1 dimension features.

    The shape of input feature should be (bs, input_c, S, d1),
        where bs, input_c, S is fixed:
            - bs: batch size
            - input_c: the init channel (could be changed)
            - S: feature length
        and [d1] can be changed, [d1] could be [F] or [L]
            - D: Direction (0-bid, 1-ask)
            - F: Feature (0-price, 1-volume, ...)

    Attention: the input features should be normed before forwarding !!!
    """

    def __init__(
            self, conv_mid_channels: List[int], dim_list: List[int],
            bn_type: str, bn_momentum: float
    ):
        """ Init of Conv2d_1Dim.

        :param conv_mid_channels: the channels of `Conv2d` layers, should have 2 int items, first one means input_c, others are mid channels
        :param dim_list: the list of 1dim, should have 1 int items, be corresponded with the encoding feature
        :param bn_type: the type of batch norm layer
        :param bn_momentum: the momentum of get_bn_class(bn_type)

        """

        super(Conv2d_1Dim, self).__init__()

        # ---- Conv 1. Conv Layer for Dim 1 ---- #
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[0], out_channels=conv_mid_channels[1], kernel_size=(1, dim_list[0]), stride=(1, dim_list[0])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[1], momentum=bn_momentum)
        )

    def forward(self, features_dim1: torch.Tensor):
        """ Forward computing of Conv2d_1Dim.

        :param features_dim1: the input features with 1 dimension, should be a 4D tensor.
            shape=(bs, input_c, S, d1), (MUST after Norm !!!).

        :return: encoding x, shape=(bs, mid_c[1], S, 1)

        """

        # ---- Conv dim 1 ---- #
        # shape from (bs, mid_c[1], S, d1) to (bs, mid_c[3], S, 1)
        x = self.conv_d1(features_dim1)
        return x


class Conv2d_3Dim_Mul(Conv2d_3Dim):
    """ Replace the first conv layer to Mul Conv2d layer.

    If you want to use this encoder, there are TWO notes you need know:
        - the input feature should be (bs, c, S, L, D, F) same as the tradition
        - the input feature need `PriceVolumeMeanNorm` in `void_norm.py`

    """

    def __init__(
            self, conv_mid_channels: List[int], dim_list: List[int], bn_type: str, bn_momentum: float,
            epsilon: float = 0.1, fixed: bool = False
    ):
        """ Init of Conv2d_3Dim_Mul. """

        super(Conv2d_3Dim_Mul, self).__init__(conv_mid_channels, dim_list, bn_type, bn_momentum)

        # ---- Conv 1. Conv Layer for Dim 3 ---- #
        self.conv_d3 = nn.Sequential(
            MultiplicativeConv2d(
                in_channels=conv_mid_channels[0], out_channels=conv_mid_channels[1], kernel_size=(1, dim_list[2]), stride=(1, dim_list[2]),
                epsilon=epsilon, fixed=fixed
            ),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[1], momentum=bn_momentum)
        )

        # ---- Conv 2. Conv Layer for Dim 2 ---- #
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[1], out_channels=conv_mid_channels[2], kernel_size=(1, dim_list[1]), stride=(1, dim_list[1])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[2], momentum=bn_momentum)
        )

        # ---- Conv 3. Conv Layer for Dim 1 ---- #
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[2], out_channels=conv_mid_channels[3], kernel_size=(1, dim_list[0]), stride=(1, dim_list[0])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[3], momentum=bn_momentum)
        )


class Conv2d_3Dim_Frac(Conv2d_3Dim):
    """ Replace the first conv layer to Frac Conv2d layer.

    If you want to use this encoder, there are TWO notes you need know:
        - the input feature should be (bs, c, S, L, D, F) same as the tradition
        - the input feature doesn't need any norm, just forward the raw data

    """

    def __init__(
            self, conv_mid_channels: List[int], dim_list: List[int],
            bn_type: str, bn_momentum: float, fixed: bool = False
    ):
        """ Init of Conv2d_3Dim_Frac. """

        super(Conv2d_3Dim_Frac, self).__init__(conv_mid_channels, dim_list, bn_type, bn_momentum)

        # ---- Conv 1. Conv Layer for Dim 3 ---- #
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[0], out_channels=conv_mid_channels[1], kernel_size=(1, dim_list[2]), stride=(1, dim_list[2])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[1], momentum=bn_momentum)
        )

        # ---- Conv 2. Conv Layer for Dim 2 ---- #
        self.conv_d2 = nn.Sequential(
            FractionalConv2d(
                in_channels=conv_mid_channels[1], out_channels=conv_mid_channels[2], kernel_size=(1, dim_list[1]), stride=(1, dim_list[1]),
                fixed=fixed
            ),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[2], momentum=bn_momentum)
        )

        # ---- Conv 3. Conv Layer for Dim 1 ---- #
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_mid_channels[2], out_channels=conv_mid_channels[3], kernel_size=(1, dim_list[0]), stride=(1, dim_list[0])),
            nn.GELU(),
            get_bn_class(bn_type)(conv_mid_channels[3], momentum=bn_momentum)
        )
