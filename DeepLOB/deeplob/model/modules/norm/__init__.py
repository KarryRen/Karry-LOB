# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/22 16:22

""" The interface of normalization. """

from torch import nn

from ..base import VoidModule
from .pv_norm import PriceVolumeFillna, FillnaStepExpAtte
from .void_norm import PriceVolumeNorm, PriceVolumeMeanNorm
from .bn_norm import SLDF_BatchNorm, SLF_BatchNorm, SDF_BatchNorm, SF_BatchNorm
from ..batch_norm import get_bn_instance


def get_norm_class(cls_kwargs: dict):
    """ Get the class of norm (useless now).

    :param cls_kwargs: the kwargs of norm class, must have `type` key

    """
    ...


def get_norm_instance(cls_kwargs: dict, init_kwargs: dict) -> nn.Module:
    """ Get the instance of norm.

    :param cls_kwargs: the kwargs of norm class
    :param init_kwargs: the kwargs of norm instance init functions

    Attention: Please be careful about the format of cls_kwargs and cls_kwargs
        - MUST have `type` key in cls_kwargs
        - MUST have `feature_list` key in init_kwargs

    """

    # ---- Get the norm type ---- #
    norm_type = cls_kwargs["type"]  # get the norm type

    # ---- Build the instance based on the `norm_type` and `init_kwargs` ---- #
    if norm_type == "PVFillna":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        fill_value = init_kwargs.get("fill_value", 0.0)  # get the fill value, default is `0.0`
        norm_instance = VoidModule(PriceVolumeFillna(fill_value=fill_value), name_list=name_list)  # build the Fillna instance
    elif norm_type == "FSExpAtte":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        init_exp_lambda = init_kwargs.get("init_exp_lambda", 1.0)  # get the init exp lambda, default is `1.0`
        norm_instance = VoidModule(FillnaStepExpAtte(init_exp_lambda=init_exp_lambda), name_list=name_list)  # build the FillnaStep Exp Atte instance
    elif norm_type == "PVNorm":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        mean_std_time_steps = init_kwargs.get("mean_std_time_steps", 100)  # get the mean_std_steps, default is `100`
        norm_instance = PriceVolumeNorm(name_list=name_list, mean_std_time_steps=mean_std_time_steps)  # build the PV Norm instance
    elif norm_type == "PVMNorm":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        mean_std_time_steps = init_kwargs.get("mean_std_time_steps", 100)  # get the mean_std_steps, default is `100`
        norm_instance = PriceVolumeMeanNorm(name_list=name_list, mean_std_time_steps=mean_std_time_steps)  # build the PVM Norm instance
    elif norm_type == "BN_3Dim":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        bn_type = init_kwargs.get("bn_type", "BN")  # get the bn_type, default is `BN`
        feature_dim = init_kwargs.get("feature_dim", 1)  # get the feature dim, default is `1`
        sldf_bn = SLDF_BatchNorm(bn=get_bn_instance(bn_type=bn_type, feature_dim=feature_dim))  # build the SLDF BN
        norm_instance = VoidModule(sldf_bn, name_list)  # build the void module instance of SLDF BN
    elif norm_type == "BN_2Dim":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        bn_type = init_kwargs.get("bn_type", "BN")  # get the bn_type, default is `BN`
        feature_dim = init_kwargs.get("feature_dim", 1)  # get the feature dim, default is `1`
        sdf_bn = SDF_BatchNorm(bn=get_bn_instance(bn_type=bn_type, feature_dim=feature_dim))  # build the SDF BN
        norm_instance = VoidModule(sdf_bn, name_list)  # build the void module instance of SDF BN
    elif norm_type == "BN_1Dim":
        name_list = init_kwargs.get("name_list", [])  # get the name_list (feature name list) , default is `[]`
        bn_type = init_kwargs.get("bn_type", "BN")  # get the bn_type, default is `BN`
        feature_dim = init_kwargs.get("feature_dim", 1)  # get the feature dim, default is `1`
        sf_bn = SF_BatchNorm(bn=get_bn_instance(bn_type=bn_type, feature_dim=feature_dim))  # build the SF BN
        norm_instance = VoidModule(sf_bn, name_list)  # build the void module instance of SF BN
    else:  # other type is not supported now
        raise TypeError(norm_type)

    return norm_instance
