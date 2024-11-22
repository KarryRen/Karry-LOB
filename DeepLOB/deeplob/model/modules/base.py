# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 2024/4/8 12:54
#
# pylint: disable=no-member

""" The Void nn.Module, `forward()` function will return None. """

from typing import List, Dict, Optional
from torch import nn
import torch


class VoidModule(nn.Module):
    def __init__(self, model: Optional[nn.Module], name_list: List[str]):
        """ Init of VoidModule.

        :param model: the model to do computation
        :param name_list: the list of name to do computation

        """

        super(VoidModule, self).__init__()
        self.model = model
        self.name_list = name_list

    def forward(self, tensor_dict: Dict[str, List[torch.Tensor]]) -> None:
        """ Forward of VoidModule. Computing the tensor in `name_list` by model.

        :param tensor_dict: the dict the raw tensors, format should be:
            {
                name_1:[raw_tensor_of_name_1, after_1_computation_of_name_1, after_2, ..., after_n],
                name_2:[raw_tensor_of_name_2, after_1_computation_of_name_2, after_2, ..., after_n],
                ...,
                name_n:[raw_tensor_of_name_n, after_1_computation_of_name_n, after_2, ..., after_n],
            }

        return: None

        """

        # ---- For loop the name_list to do the computation ---- #
        for name in self.name_list:  # not all name in tensor_dict will be computed
            # append the computing result and keep the raw data
            tensor_dict[name].append(self.model(tensor_dict[name][-1]))

        # ---- Return None ---- #
        return
