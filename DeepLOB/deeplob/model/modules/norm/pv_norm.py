# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The price volume norm class.
    - PriceVolumeFillna: The Price Volume Fillna norm class.

"""

import torch
from torch import nn
import math


class PriceVolumeFillna(nn.Module):
    """ The Price Volume Fillna class. Will fill the nan in price and volume, when use (S, PN, D, F) features.

    Attention: based on the Tensor Engineering Algorithm,
        - the raw `nan` in Volume is `-10.0`
        - after the Exp Attenuation operation, the `nan` in Volume is where < 0.0

    """

    def __init__(self, fill_value: float = 0.0):
        """ Init of PriceVolumeFillna.

        :param fill_value: the value to fill nan

        """

        super(PriceVolumeFillna, self).__init__()

        self.fill_value = fill_value
        self.register_parameter("dummy", None)  # no learnable params

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Forward of PriceVolumeFillna.

        :param tensor: the input tensor will be fill nan

        return:
            - tensor_fillna: the tensor after fill nan

        """

        tensor_fillna = tensor.clone()  # clone to init the tensor_fillna (MUST !!!)
        nan_index = torch.where(tensor_fillna < 0.0)  # get the nan index
        tensor_fillna[nan_index] = self.fill_value  # fill nan
        assert (tensor_fillna >= 0).all(), "PriceVolumeFillna ERROR !!"  # test whether fill all nan
        return tensor_fillna


class FillnaStepExpAtte(nn.Module):
    """ Use the FillnaStep to do the Exponential Attenuation of Volume, when use (S, PN, D, F) features.

    `ExpAtte_Volume = Volume * Exp(-exp_lambda*FillnaStep)`
        you can see the `log` and `exp` operations, which in order to keep exp_lambda is > 0.

    Attention: this norm way only supports 3 Features now, which are 0-Price, 1-Volume, 2-FillnaStep.

    """

    def __init__(self, init_exp_lambda: float):
        """ Init of FillnaStepExpAtte.

        :param init_exp_lambda: the init value of exp lambda, must > 0

        """

        super(FillnaStepExpAtte, self).__init__()

        assert init_exp_lambda > 0, "The `init_exp_lambda` of FillnaStepExpAtte Norm Module should > 0 !!!"
        self.exp_lambda = nn.Parameter(torch.tensor([math.log(init_exp_lambda)]))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Forward of FillnaStepExpAtte.

        :param tensor: the input tensor to be Exp Atte, the F dim should be the last dim, where:
            0-Price
            1-Volume
            2-FillnaStep

        return:
            - tensor_exp_atte_pv: the tensor after Exp Atte, only have P&B

        """

        # ---- Step 1. Test the last dim of tensor is 3 ---- #
        assert tensor.shape[-1] == 3

        # ---- Step 2. Clone to init the tensor_exp_atte (MUST !!!) ---- #
        tensor_exp_atte_pv = tensor[..., :2].clone()  # only get P & V, the last dim of tensor is 2

        # ---- Step 3. Do the Exp Atte and Return ---- #
        tensor_exp_atte_pv[..., 1] = tensor[..., 1] * torch.exp(-torch.matmul(tensor[..., 2:], torch.exp(self.exp_lambda)))
        return tensor_exp_atte_pv
