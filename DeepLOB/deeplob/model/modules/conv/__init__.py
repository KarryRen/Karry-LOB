# -*- coding: utf-8 -*-
# @author : MaMing, RenKai (intern in HIGGS ASSET)
# @time   : 4/24/24 10:05 AM
#
# pylint: disable=no-member

""" The interfaces of `Conv2d`. """

from torch import nn

from .conv2d import Conv2d_3Dim
from .conv2d import Conv2d_2Dim
from .conv2d import Conv2d_1Dim
from .conv2d import Conv2d_3Dim_Mul
from .conv2d import Conv2d_3Dim_Frac


def get_conv2d_class(cls_kwargs: dict):
    """ Get the class of `Conv2d` based on the `cls_kwargs`.

    :param cls_kwargs: the kwargs of `Conv2d` class, MUST have `type` key

    return: the `Conv2d` class.

    """

    # ---- Get the conv2d type ---- #
    conv2d_type = cls_kwargs["type"]  # get the type

    # ---- Build the class based on the `norm_type` and `init_kwargs` ---- #
    if conv2d_type == "Conv2d_3Dim":
        return Conv2d_3Dim
    elif conv2d_type == "Conv2d_2Dim":
        return Conv2d_2Dim
    elif conv2d_type == "Conv2d_1Dim":
        return Conv2d_1Dim
    elif conv2d_type == "Conv2d_3Dim_Mul":
        return Conv2d_3Dim_Mul
    elif conv2d_type == "Conv2d_3Dim_Frac":
        return Conv2d_3Dim_Frac
    else:
        raise TypeError(conv2d_type)


def get_conv2d_instance(cls_kwargs: dict, init_kwargs: dict) -> nn.Module:
    """ Get the instance of `Conv2d` based on the `cls_kwargs` and `init_kwargs`.

    :param cls_kwargs: the kwargs of `Conv2d` class
    :param init_kwargs: the kwargs of `Conv2d` instance init functions

    return: the `Conv2d` instance

    Attention: Please be careful about the format of cls_kwargs and cls_kwargs
        - cls_kwargs MUST have `type` key !
    """

    # ---- Get the `Conv2d` instance and return ---- #
    conv2d_instance = get_conv2d_class(cls_kwargs=cls_kwargs)(**init_kwargs)
    return conv2d_instance
