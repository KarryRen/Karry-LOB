# -*- coding: utf-8 -*-
# @author : Maming, RenKai (intern in HIGGS ASSET)
# @time   : 2024/03/30 9:08
#
# pylint: disable=no-member

""" The norm class that returns None during forward (the reason why we call this file `void_norm.py`):
    - PriceVolumeNorm: Price, Volume(logged) normalization of (bs, init_channels, time_steps, 2 directions, 5 levels) data
        `x = (x - mean(x)) / std(x)`
    - PriceVolumeMeanNorm
        `x = x / mean(x)`

"""

import torch
from typing import List, Dict

from ..base import VoidModule


class PriceVolumeNorm(VoidModule):
    """ Price, Volume(logged) normalization of (bs, init_channels(might not feature_len), time_steps, 2 directions, 5 levels) data.
    This norm way can keep the price high and low relationship, shift the distribution to N(0, 1) by: `x = (x - mean(x)) / std(x)`

    Attention:
        - PriceVolumeNorm just compute the mean&std ONCE, using the FIRST name of tensor_dict !
        - Please be careful about the tensor shape in tensor dict. The shape of tensor should
            be: (bs, init_channel, S, L, D, F) and for `F` index 0 MUST BE PRICE and the index 1 MUST BE VOLUME !

    """

    def __init__(self, name_list: List[str], mean_std_time_steps: int):
        """ Init of PriceVolumeNorm.

        :param name_list: the list of name to do computation
        :param mean_std_time_steps: the number of time steps to compute the mean & std

        """

        super(PriceVolumeNorm, self).__init__(None, name_list)

        self.mean_std_time_steps = mean_std_time_steps
        self.register_parameter("dummy", None)

    def forward(self, tensor_dict: Dict[str, List[torch.Tensor]]) -> None:
        """ Forward of PriceVolumeNorm. PVNorm the last item tensor in `name_list`.

        :param tensor_dict: the dict the raw tensors, format should be:
            {
                name_1:[raw_tensor_of_name_1, after_1_computation_of_name_1, after_2, ..., after_n],
                name_2:[raw_tensor_of_name_2, after_1_computation_of_name_2, after_2, ..., after_n],
                ...,
                name_n:[raw_tensor_of_name_n, after_1_computation_of_name_n, after_2, ..., after_n]
            }


        return: None

        """

        # ---- The first name flag ---- #
        is_first_name = True

        # ---- For loop to norm ---- #
        for name in self.name_list:  # not all name in tensor_dict will be computed
            # get the last item and `clone` it to init the x_fillna (MUST !!!)
            norm_tensor = tensor_dict[name][-1].clone()
            # log the volume, which means the
            norm_tensor[..., 1] = torch.log(norm_tensor[..., 1] + 1)
            # compute the mean and std of the first tensor of (S, L, D) dims
            if is_first_name:
                mean = norm_tensor[:, :, -self.mean_std_time_steps:, :, :, :].mean(dim=(2, 3, 4), keepdim=True).detach()  # shape = (bs, 1, 1, 1, 1, f)
                std = norm_tensor[:, :, -self.mean_std_time_steps:, :, :, :].std(dim=(2, 3, 4), keepdim=True).detach() + 1e-5  # shape = (bs, 1, 1, 1, 1, f)
                is_first_name = False  # only compute the mean&std of the first feature
            # do the z-score norm (just P&V)
            norm_tensor[..., :2] = (norm_tensor[..., :2] - mean[..., :2]) / std[..., :2]
            # append the `norm_tensor` to tensor_dict
            tensor_dict[name].append(norm_tensor)

        # ---- Return None ---- #
        return


class PriceVolumeMeanNorm(VoidModule):
    """ Price, Volume(logged) normalization of (bs, init_channels(might not feature_len), time_steps, 2 directions, 5 levels) data.
    This norm way can keep the price high and low relationship, shift the distribution to (1, std) by: `x = x / mean(x)`

    Attention:
        - PriceVolumeMeanNorm just compute the mean ONCE, using the FIRST name of tensor_dict !
        - Please be careful about the tensor shape in tensor dict. The shape of tensor should
            be: (bs, init_channel, S, L, D, F) and for `F` index 0 MUST BE PRICE and the index 1 MUST BE VOLUME !
        - PriceVolumeMeanNorm is special for `MultiplicativeConv2d`

    """

    def __init__(self, name_list: List[str], mean_std_time_steps: int):
        """ Init of PriceVolumeMeanNorm.

        :param name_list: the list of name to do computation
        :param mean_std_time_steps: the number of time steps to compute the mean & std

        """

        super(PriceVolumeMeanNorm, self).__init__(None, name_list)

        self.mean_std_time_steps = mean_std_time_steps
        self.register_parameter("dummy", None)

    def forward(self, tensor_dict: Dict[str, List[torch.Tensor]]) -> None:
        """ Forward of PriceVolumeMeanNorm. PVMNorm the tensor in `name_list`.

        :param tensor_dict: the dict the raw tensors, format should be:
            {
                name_1:[raw_tensor_of_name_1, after_1_computation_of_name_1, after_2, ..., after_n],
                name_2:[raw_tensor_of_name_2, after_1_computation_of_name_2, after_2, ..., after_n],
                ...,
                name_n:[raw_tensor_of_name_n, after_1_computation_of_name_n, after_2, ..., after_n],
            }


        return: None

        """

        # ---- The first name flag ---- #
        is_first_name = True

        # ---- For loop to norm ---- #
        for name in self.name_list:  # not all name in tensor_dict will be computed
            # get the last item and `clone` it to init the x_fillna (MUST !!!)
            norm_tensor = tensor_dict[name][-1].clone()
            # log the volume, which means the
            norm_tensor[..., 1] = torch.log(norm_tensor[..., 1] + 1)
            # compute the mean and std of the first tensor of (S, L, D) dims
            if is_first_name:
                mean = norm_tensor[:, :, -self.mean_std_time_steps:, :, :, :].mean(dim=(2, 3, 4), keepdim=True).detach() + 1e-5  # shape = (bs, 1, 1, 1, 1, f)
                is_first_name = False  # only compute the mean&std of the first feature
            # do the PVM norm operation
            norm_tensor[..., :2] = norm_tensor[..., :2] / mean[..., :2]
            # append the `norm_tensor` to tensor_dict
            tensor_dict[name].append(norm_tensor)

        # ---- Return None ---- #
        return
