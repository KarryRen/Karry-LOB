# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/22 15:16

""" The interface of deep lob net. """

from typing import Union
import torch
from torch import nn

from .net import DeepLOBNet


def get_instance_of_net(device: torch.device, codes: Union[list, str] = None, **net_config) -> nn.Module:
    """ Get the instance of net，

    :param device: the computing device
    :param codes: the codes to be modeled (now only support one code)
    :param net_config: other configs about the net structure (see the detail in config)

    :return:
        - net_instance: the instance of net, all weights are determined by the `init_model_seed`

    """

    # ---- Build up the net instance ---- #
    if isinstance(codes, str):  # model for single code
        net_instance = DeepLOBNet(device=device, **net_config)
    else:
        raise TypeError(codes)

    # ---- Return the net instance ---- #
    return net_instance
