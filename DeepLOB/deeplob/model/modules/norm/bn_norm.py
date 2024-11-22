# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 4/23/24 1:12 PM
#
# pylint: disable=no-member

""" The bn norm class. We call these classes as bn_norm because we want to separate them from the BN in torch.nn.
    - SF_BatchNorm: for `(S, F)` type features
    - `SLF_BatchNorm`: for `(S, L, F)` type features
    - `SDF_BatchNorm`: for `(S, D, F)` type features
    - `SLDF_BatchNorm`: for `(S, L, D, F)` type features

"""

import torch
from torch import nn


class SF_BatchNorm(nn.Module):
    """ The batch norm for `(S, F)` features. Such as `voi (volume order imbalance)`.

    The shape of input feature is excepted to be (bs, 1, S, F)
        - bs: batch size
        - 1: the init channel (could be changed)
        - S: feature length
        - F: Features, index is based on the (S, F) features sequence in config file

    """

    def __init__(self, bn: nn.Module):
        """ Init of `(S, F)` batch norm.

        :param bn: the batch norm module

        """

        super(SF_BatchNorm, self).__init__()
        self.bn = bn

    def forward(self, sf_features: torch.Tensor):
        """ Forward of `(S, F)` batch norm.

        :param sf_features: the input `(S, F)` features, a 4D tensor. shape=(bs, 1, S, F)

        return:
            - normed_sf_features, the normed `(S, F)` features, a 4D tensor. shape=(bs, 1, S, F)

        """

        # ---- Step 1. Get feature shape ---- #
        bs, input_c, S, F = sf_features.shape

        # ---- Step 2. BN for Normalization (by F dim) ---- #
        # transpose, shape from (bs, input_c, S, F) to (bs, input_c, F, S)
        sf_features = sf_features.permute(0, 1, 3, 2)
        # reshape, shape from (bs, input_c, F, S) to (bs, input_c*F, S, 1)
        sf_features = torch.reshape(sf_features, (bs, input_c * F, S, 1))
        # BN in F dim, shape=(bs, input_c*F, S, 1)
        normed_sf_features = self.bn(sf_features)
        # reshape back, shape from (bs, input_c*F, S, 1) to (bs, input_c, F, S)
        normed_sf_features = torch.reshape(normed_sf_features, (bs, input_c, F, S))
        # transpose back, shape from (bs, input_c, F, S) to (bs, input_c, S, F)
        normed_sf_features = normed_sf_features.permute(0, 1, 3, 2)

        # ---- Step 3. Return the normed sf_feature ---- #
        return normed_sf_features


class SLF_BatchNorm(nn.Module):
    """ The batch norm for `(S, L, F)` features. Such as `pi_vi`.

    The shape of input feature is excepted to be (bs, 1, S, L, F)
        - bs: batch size
        - 1: the init channel (could be changed)
        - S: feature length
        - L: Level (0 ~ 4 represent `level 1 to level 5`)
        - F: Features, index is based on the (S, L, F) features sequence in config file

    """

    def __init__(self, bn: nn.Module):
        """ Init of `(S, L, F)` features encoder.

        :param bn: the batch norm module

        """

        super(SLF_BatchNorm, self).__init__()
        self.bn = bn

    def forward(self, slf_features: torch.Tensor):
        """ Forward of `(S, L, F)` batch norm.

        :param slf_features: the input `(S, L, F)` features, a 5D tensor, shape=(bs, 1, S, L, F)

        return:
            - normed_slf_features, the normed `(S, L, F)` features, a 5D tensor, shape=(bs, 1, S, L, F)

        """

        # ---- Step 1. Get feature shape ---- #
        bs, input_c, S, L, F = slf_features.shape

        # ---- Step 2. BN for Normalization (by F dim) ---- #
        # transpose, shape from (bs, input_c, S, L, F) to (bs, input_c, F, S, L)
        slf_features = slf_features.permute(0, 1, 4, 2, 3)
        # reshape, shape from (bs, input_c, F, S, L) to (bs, input_c*F, S, L)
        slf_features = torch.reshape(slf_features, (bs, input_c * F, S, L))
        # BN in F dim, shape=(bs, input_c*F, S, L)
        normed_slf_features = self.bn(slf_features)
        # reshape back, shape from (bs, input_c*F, S, L) to (bs, input_c, F, S, L)
        normed_slf_features = torch.reshape(normed_slf_features, (bs, input_c, F, S, L))
        # transpose back, shape from (bs, input_c, F, S, L) to (bs, input_c, S, L, F)
        normed_slf_features = normed_slf_features.permute(0, 1, 3, 4, 2)

        # ---- Step 3. Return the normed slf_feature ---- #
        return normed_slf_features


class SDF_BatchNorm(nn.Module):
    """ The batch norm for `(S, D, F)` features. Such as `order_increase`.

    The shape of input feature is excepted to be (bs, 1, S, D, F)
        - bs: batch size
        - 1: the init channel (could be changed)
        - S: feature length
        - D: Direction (0-bid, 1-ask)
        - F: Feature

    Attention: SDF is SAME as SLF actually !

    """

    def __init__(self, bn: nn.Module):
        """ Init of `(S, D, F)` batch norm.

        :param bn: the batch norm module

        """

        super(SDF_BatchNorm, self).__init__()
        self.bn = bn

    def forward(self, sdf_features: torch.Tensor):
        """ Forward of `(S, D, F)` batch norm.

        :param sdf_features: the input `(S, D, F)` features, a 5D tensor. shape=(bs, 1, S, D, F)

        return:
            - normed_sdf_features, the normed `(S, D, F)` features, a 5D tensor. shape=(bs, 1, S, D, F)

        """

        # ---- Step 1. Get feature shape ---- #
        bs, input_c, S, D, F = sdf_features.shape

        # ---- Step 2. BN for Normalization (by F dim) ---- #
        # transpose, shape from (bs, input_c, S, D, F) to (bs, input_c, F, S, D)
        sdf_features = sdf_features.permute(0, 1, 4, 2, 3)
        # reshape, shape from (bs, input_c, F, S, D) to (bs, input_c*F, S, D)
        sdf_features = torch.reshape(sdf_features, (bs, input_c * F, S, D))
        # BN in F dim, shape=(bs, input_c*F, S, D)
        normed_sdf_features = self.bn(sdf_features)
        # reshape back, shape from (bs, input_c*F, S, D) to (bs, input_c, F, S, D)
        normed_sdf_features = torch.reshape(normed_sdf_features, (bs, input_c, F, S, D))
        # transpose back, shape from (bs, input_c, F, S, D) to (bs, input_c, S, D, F)
        normed_sdf_features = normed_sdf_features.permute(0, 1, 3, 4, 2)

        # ---- Step 3. Return the normed sf_feature ---- #
        return normed_sdf_features


class SLDF_BatchNorm(nn.Module):
    """ The batch norm for `(S, L, D F)` features. Such as `loi (leveled order increase)`.

    The shape of input feature is excepted to be (bs, 1, S, L, D, F)
        - bs: batch size
        - 1: the init channel (could be changed)
        - S: feature length
        - L: Level (0 ~ 4 represent `level 1 to level 5`)
        - D: Direction (0-bid, 1-ask)
        - F: Features, index is based on the (S, F) features sequence in config file

    """

    def __init__(self, bn: nn.Module):
        """ Init of `(S, L, D, F)` batch norm.

        :param bn: the batch norm module

        """

        super(SLDF_BatchNorm, self).__init__()
        self.bn = bn

    def forward(self, sldf_features: torch.Tensor):
        """ Forward computing of `(S, L, D, F)` Features Encoder.

        :param sldf_features: the input `(S, L, D, F)` features, a 6D tensor, shape=(bs, 1, S, L, D, F)

        return:
            - normed_sldf_features, the normed `(S, L, D, F)` features, a 6D tensor, shape=(bs, 1, S, L, D, F)

        """

        # ---- Step 1. Get feature shape ---- #
        bs, input_c, S, L, D, F = sldf_features.shape

        # ---- Step 2. BN for Normalization (by F dim) ---- #
        # transpose, shape from (bs, input_c, S, L, D, F) to (bs, input_c, F, S, L, D)
        sldf_features = sldf_features.permute(0, 1, 5, 2, 3, 4)
        # reshape, shape from (bs, input_c, F, S, L, D) to (bs, input_c*F, S, L, D)
        sldf_features = torch.reshape(sldf_features, (bs, input_c * F, S, L * D))
        # BN in F dim, shape=(bs, input_c*F, S, L, D)
        normed_sldf_features = self.bn(sldf_features)
        # reshape back, shape from (bs, input_c*F, S, L, D) to (bs, input_c, F, S, L, D)
        normed_sldf_features = torch.reshape(normed_sldf_features, (bs, input_c, F, S, L, D))
        # transpose back, shape from (bs, input_c, F, S, L, D) to (bs, input_c, S, L, D, F)
        normed_sldf_features = normed_sldf_features.permute(0, 1, 3, 4, 5, 2)

        # ---- Step 3. Return the normed sf_feature ---- #
        return normed_sldf_features
