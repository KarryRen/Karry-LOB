# -*- coding: utf-8 -*-
# @author : MaMing, RenKai (intern in HIGGS ASSET)
# @time   : 4/1/24 12:02 PM
#
# pylint: disable=no-member

""" The interface of modules. """

from typing import Union
import torch
from torch import nn

from .net import DeepLOBNet
from .net import DeepLOBNetMultiCodes


def get_net_class_single_code(class_type: str = None, codes: Union[list, str] = None):
    """ The net class loader for single code. """
    return DeepLOBNet


def get_net_class_multi_codes(class_type: str = None, codes: Union[list, str] = None):
    """ The net class loader for multi codes. """
    return DeepLOBNetMultiCodes


def get_instance_of_net(device: torch.device, class_type: str = None, codes: Union[list, str] = None, **net_config) -> nn.Module:
    """ Get the instance of net，

    :param device: the computing device
    :param class_type: the type of class
    :param codes: the code to be modeled
    :param net_config: other config about the net structure

    return:
        - net_instance: the instance of net, all weights are determined by the `init_model_seed`
    """

    # ---- Build up the net instance ---- #
    if isinstance(codes, str):  # model for single code
        net_class = get_net_class_single_code(class_type, codes)
        net_instance = net_class(device=device, **net_config)
    elif isinstance(codes, list):  # model for multi codes
        net_class = get_net_class_multi_codes(class_type, codes)
        net_instance = net_class(device=device, codes=codes, **net_config)
    else:
        raise TypeError(codes)

    # ---- Return the net instance ---- #
    return net_instance
