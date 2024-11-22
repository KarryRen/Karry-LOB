# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 6/2/24 10:47 AM
#
# pylint: disable=no-member

""" The diy conv layers, just one layer rather than a module !!

    - MultiplicativeConv2d
    - FractionalConv2d

"""

from typing import Tuple
import torch
import warnings
from torch import nn


class MultiplicativeConv2d(nn.Module):
    """ Applies a Multiplicative 2D convolution over an input signal composed of several input
    planes. This is specifically designed for scenarios where multiplication may occur.

    For example, we might need `price * volume` in factor recognition.
    Take this as the simplest case, given a tensor X = [pb_1, vb_1].
    When use the traditional Conv2d with a kernel whose size is (1, 2) we can get feature: `Y = w_1*pb_1 + w_2*vb_1 + b`
    When use our MultiplicativeConv2d with a kernel whose size is (1, 2) we can get feature: `Y = e^b * (pb_1^w_1 * vb_1*w_2)`

    To realize this operation, we add `log` and `exp` operations before and after the
    convolution, which means we will do three-step operations:
        - Step 1. log_x = log(x)
        - Step 2. con_log_x = conv(log_x)
        - Step 3. y = exp(con_log_x)

    Note:
        1. The input `x` can have the <= 0 value, but we will clip `x` to [`epsilon` to inf).
           Because when `x <= 0` this operation is meaningless:
                (1) If transpose the negative value to positive, it will lose the big & small relationship.
                (2) -1^(2.3) will generate imaginary number.
           If the negative or small positive value is really important, we WILL NOT suggest you to use
           the module !!!!
        2. The `epsilon` CAN'T be so close to 0, to avoid the gradient explosion.
           And 0.1 might be a good choice.
        3. The parameters that can be learned in this module are those in the convolution
           operation.
        4. There is a `fixed` param, if False just like the top description. if True y = w*p*v

    """

    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: Tuple[int, int], stride: Tuple[int, int], epsilon: float = 0.1,
            fixed: bool = False
    ):
        """ Initialization of the `MultiplicativeConv2d` module.

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param kernel_size: size of the conv kernel
        :param stride: stride of the convolution
        :param epsilon: the target value of replaced <= 0 value
        :param fixed: fix do the multiplication or not

        """

        super(MultiplicativeConv2d, self).__init__()

        # ---- Params ---- #
        self.epsilon = epsilon
        self.fixed = fixed

        # ---- Fixed or not have different way to init conv module ---- #
        if self.fixed:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
            # init weight & bias
            nn.init.normal_(self.conv.weight, mean=1, std=0.1)
            nn.init.constant_(self.conv.bias, val=0)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            # init weight & bias
            # - the weight (have two ways)
            # ~ way 1: N(1, 0.1)
            # nn.init.normal_(self.conv.weight, mean=1, std=0.1)
            # ~ way 2: B(-1, 1)
            zero_one_dis = (torch.rand(16, 1, 1, 2) > 0.5) * 2 - 1
            zero_one_dis_param = nn.Parameter(zero_one_dis * torch.normal(1, 0.1, size=(16, 1, 1, 2)))
            self.conv.weight = zero_one_dis_param
            # - the bias
            nn.init.constant_(self.conv.bias, val=0)

    def forward(self, x: torch.Tensor):
        """ Forward computing of MultiplicativeConv2d.

        :param x: the input features, MUST be a 4D tensor. shape=(bs, input_c, S, f)

        :return: encoding x, shape=(bs, output_c, S, f//2)

        """

        # ---- Check the min of x ---- #
        if x.min() < self.epsilon:
            warnings.warn(
                f"ATTENTION PLEASE ! The input of module `{self._get_name()}` "
                f"have values < `epsilon` you set ({self.epsilon}) and will be clip to "
                f"[epsilon, +inf). If the clipping is meaningless for you, please "
                f"NEVER use this module !"
            )

        # ---- Clip x to [epsilon, inf) ---- #
        x = torch.clamp(x, min=self.epsilon)

        # ---- Mul operation ---- #
        if self.fixed:
            # directed mul operation
            p_mul_v = x[:, :, :, 0::2] * x[:, :, :, 1::2]
            output = self.conv(p_mul_v)
        else:
            # three-steps operation
            log_x = torch.log(x)
            con_log_x = self.conv(log_x)
            output = torch.exp(con_log_x)
        return output


class FractionalConv2d(nn.Module):
    """ Applies a FractionalConv2d 2D convolution over an input signal composed of several input
    planes. This is specifically designed for scenarios where fraction may occur.

    For example, we might need `(ask - bid) / (ask + bid)` in factor recognition. Take this as the
    simplest case, given a tensor X = [b_1, a_1].
    When use the traditional Conv2d with a kernel whose size is (1, 2) we can get feature: `Y = w_1*b_1 + w_2*a_1 + b`
    When use our FractionalConv2d with a kernel whose size is (1, 2) we can get feature: `Y = (ask - w * bid + b) / (ask + w * bid + b)`

    Note:
        1. The parameters that can be learned in this module are those in the convolution operation.
        2. Init the w_1 = 1, b_1 = 0 and freeze it !
        3. There is a `fixed` param, if False just like the top description. if True y = w*(a - b) / (a + b)

    """

    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: Tuple[int, int], stride: Tuple[int, int], fixed: bool = False
    ):
        """ Initialization of the `FractionalConv2d` module.

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param kernel_size: size of the conv kernel
        :param stride: stride of the convolution

        """

        super(FractionalConv2d, self).__init__()

        # ---- Params ---- #
        self.fixed = fixed

        # ---- Fixed or not ---- #
        if self.fixed:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
            nn.init.normal_(self.conv.weight, mean=1, std=0.1)
            nn.init.constant_(self.conv.bias, val=0)
        else:
            # the freeze conv, s x 1 conv, no bias
            self.conv_freeze = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size[0], 1), stride=(stride[0], 1), bias=False
            )
            nn.init.constant_(self.conv_freeze.weight, val=1)
            self.conv_freeze.weight.requires_grad = False  # freeze it, just using for shape transforming
            # the other conv, s x (k - 1) conv, have bias
            self.conv_other = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size[0], kernel_size[1] - 1), stride=(stride[0], stride[1] - 1),
                bias=False
            )
            nn.init.normal_(self.conv_other.weight, mean=1, std=0.1)
            # nn.init.constant_(self.conv_other.bias, val=0)

    def forward(self, x: torch.Tensor):
        """ Forward computing of FractionalConv2d.

        :param x: the input features, MUST be a 4D tensor. shape=(bs, input_c, S, f)

        :return: encoding x, shape=(bs, output_c, S, f//2)

        """

        # ---- Check the x ---- #
        assert x.shape[-1] % 2 == 0, (
            f"ATTENTION PLEASE ! The last dim of input in module `{self._get_name()}` "
            f"must be even, but now is {x.shape[-1]}"
        )
        bs, s, l, d, f = x.shape[0], x.shape[2], 5, 2, 1

        # ---- Do the Fractional operation ---- #
        if self.fixed:
            numerator = x[:, :, :, 1::2] - x[:, :, :, 0::2]
            denominator = x[:, :, :, 1::2] + x[:, :, :, 0::2]
            frac_result = numerator / (denominator + 1e-5)
        else:
            numerator = self.conv_freeze(x[:, :, :, 1::2]) - self.conv_other(x[:, :, :, 0::2])
            denominator = self.conv_freeze(x[:, :, :, 1::2]) + self.conv_other(x[:, :, :, 0::2])
            frac_result = numerator / (denominator + 1e-5)

        # ---- Norm Distribution ---- #
        c = frac_result.shape[1]
        frac_result = torch.reshape(frac_result, (bs, c, s, l, f))
        frac_result_mean = frac_result.mean(dim=(2, 3), keepdim=True)
        frac_result_std = frac_result.std(dim=(2, 3), keepdim=True) + 1e-5
        frac_result = (frac_result - frac_result_mean) / frac_result_std
        frac_result = torch.reshape(frac_result, (bs, c, s, l * f))

        # ---- Set Value ---- #
        if self.fixed:
            output = self.conv(frac_result)
        else:
            output = frac_result
        return output
