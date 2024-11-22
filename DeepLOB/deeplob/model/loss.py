# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The loss function of training DeepLOB. """

import torch
from torch import nn


class MTL:
    """ Compute the multi_tick_mse_loss. """

    def __init__(self, reduction: str = "mean", ones_weight: bool = False):
        """ Init function of MTL.

        :param reduction: the reduction way
        :param ones_weight: use the ones weight or not

        """

        self.reduction = reduction
        self.ones_weight = ones_weight
        assert self.reduction in ["mean", "sum"], "We only support `mean` or `sum` reduction way now !"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor, **kwargs):
        """ Compute the deep lob loss. The weighted multi tick MSE loss.

        :param y_pred: the prediction, shape=(bs, label_len, 1) or (bs, code, label_len, 1)
        :param y_true: the label, shape=(bs, label_len, 1) or (bs, code, label_len, 1)
        :param weight: the weight, shape=(bs, 1)
        :param device: the computing device

        return:
            - batch_loss, a number, shape=()

        """

        # ---- Step 0. Get the shape&device and construct mtl_weight ---- #
        if len(y_pred.shape) == 3:
            bs, label_len, _ = y_pred.shape
        else:
            bs, code_len, label_len, _ = y_pred.shape
        device = y_pred.device
        mtl_weight = torch.softmax(torch.arange(0.0, label_len) * 2, dim=0).to(device=device)

        # ---- Step 1. Test the weight shape & make the default weight ---- #
        if weight is None:
            weight = torch.ones((bs, label_len)).to(device=device)
        else:
            assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 2. Compute the batch loss ---- #
        if self.reduction == "mean":
            batch_loss = torch.sum(weight * torch.sum((y_pred - y_true) ** 2 * mtl_weight, dim=1, keepdim=True)) / torch.sum(weight)
        elif self.reduction == "sum":
            if self.ones_weight:
                batch_loss = torch.sum((y_pred - y_true) ** 2)
            else:
                batch_loss = torch.sum((weight * (y_pred - y_true)) ** 2)
        else:
            batch_loss = 0.0
        return batch_loss


def get_loss_instance(loss_dict: dict):
    """ Get the instance of loss for training DeepLOB.

    :param loss_dict : the loss config dict, format should be {loss_type: loss_params}
        Now only support 2 types of loss:
            - `MTL`: for multi-tick-MSE loss

    """

    assert len(loss_dict) == 1, "Loss dict should have ONLY ONE item !"
    for loss_type, loss_params in loss_dict.items():
        if loss_type == "MTL":
            return MTL(**loss_params)
        else:
            raise ValueError(loss_type)
