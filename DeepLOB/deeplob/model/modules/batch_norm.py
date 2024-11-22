# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 2023/11/30 9:08
#
# pylint: disable=no-member

""" The batch norm Modules of deep-lob net.

    - bn_class_loader: the interface of batch norm modules.

"""

import torch
from torch import nn


def get_bn_class(bn_type):
    """ Get the bn class. """

    if bn_type == "BN":
        return nn.BatchNorm2d
    elif bn_type == "GBN":
        return GlobalBatchNorm2d
    else:
        raise ValueError(bn_type)


def get_bn_instance(bn_type, feature_dim, **kwargs):
    """ Get the bn instance. """

    return get_bn_class(bn_type)(feature_dim, **kwargs)


class MyBatchNorm2d(nn.BatchNorm2d):
    """
    https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L62
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class GlobalBatchNorm2d(nn.BatchNorm2d):
    """ The Global Batch Normalization for 2D feature.
    Ref. https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L62

    The core difference between this module and torch.BN are 2 points:
        - When `momentum=None`, the cumulative moving average is no longer updated every BATCH, but every EPOCH.
            The `Global` means that one round of traversal of the entire dataset is completed during the train (1 EPOCH).
        - The mean&var of training is not computed by 1 batch but by the traced mean&var, we hope this to make train ROBUSTER.

    The following train&eval structure should be strictly followed when using this module:
        for epoch in range(EPOCH):
            # start 1 `Global`
            model.train()
            train Net (the GlobalBatchNorm2d is used in Net)
            model.eval()
            eval Net (the GlobalBatchNorm2d is used in Net)
            # end 1 `Global`

    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = None, affine: bool = False, track_running_stats: bool = True):
        """ Init of the GlobalBatchNorm2d.

        All params are same as nn.BatchNorm2d, you can check the document to look the detail.

        :param num_features: the `c` from an expected input of shape (bs, c, h, w)
        :param eps: a value added to the denominator for numerical stability
        :param momentum: the value used for the running_mean and running_var computation.
            Default is set to `None` for cumulative moving average by each epoch.
        :param affine: a boolean value that when set to `True`, this module has learnable affine parameters `weight` and `bias`.
        :param track_running_stats: a boolean value that when set to `True`, this module tracks the running mean and variance.
            In this module, this param MUST be `True` !!

        """

        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.forward_status = 0  # 1 for `train`, 0 for `eval`
        self.num_epoch_tracked = 0

    def forward(self, input: torch.Tensor):
        """ Forward of the GlobalBatchNorm2d.

        :param input: the input tensor, shape=(bs, c, h, w)

        """

        # ---- Step 0. Check the input dim and do the init ---- #
        self._check_input_dim(input)
        exponential_average_factor = 0.0  # init the weight of exponential_average as 0.0

        # ---- Step 1. Update the forward status and epoch num ---- #
        if self.training:
            if self.forward_status == 0:  # first forward at `train` status
                self.num_epoch_tracked += 1
                self.forward_status = 1
        else:  # change to `eval` status
            self.forward_status = 0

        # ---- Step 2. Compute the `exponential_average_factor` ---- #
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_epoch_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # ---- Step 3. Calculate running estimates `mean` & `var`, both shape is (c) ---- #
        if self.training:  # when training, will calculate the `mean` & `var` and update the running_mean&var
            if self.num_epoch_tracked > 1:  # after the first epoch
                # mean and var are traced mean and var
                mean = self.running_mean
                var = self.running_var
                # compute the mean and var of the batch data and update the traced mean and var
                temp_mean = input.mean([0, 2, 3])
                temp_var = input.var([0, 2, 3], unbiased=False)  # use biased var in train
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * temp_mean + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * temp_var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
            else:  # the first epoch
                # mean and var are computed by batch data
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3], unbiased=False)  # use biased var in train
                # update the traced mean and var
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:  # when eval, read the running_mean&var as `mean` & `var`
            mean = self.running_mean
            var = self.running_var

        # ---- Step 4. Do the batch normalization (None is used to expand dimension) ---- #
        # z-score
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        # do the affine
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input
