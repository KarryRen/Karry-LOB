# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 1/26/24 9:25 AM
#
# pylint: disable=no-member

""" DeepLOBDataset_DailySample for deep-lob net.

We suggest you use `DeepLOBDataset_DailySample` during train for less random and more robustness

"""

from typing import List, Dict
import numpy as np
import torch
from .base import DailySampleBase


class DeepLOBDataset_DailySample(DailySampleBase):
    """ Characterizes the daily sample dataset of deep-lob for PyTorch.
        This is the new type Dataset, get (days * tick_num_in_one_sample) tick item each time.
        Drop the meaningless data, set all weights be 1.

    """

    def __len__(self) -> int:
        """ The total number of tick items. All possible items index. """

        return self.tick_gap

    def __getitem__(self, idx) -> dict:
        """ Get the item based one idx, and lag the item.

        return: item_data, (one tick of one future)
            - `features`: all needed features
            - `label`: the return label, shape=(sample_num*tick_num_in_one_sample, 1)
            - `label`: the weight, shape=(sample_num*tick_num_in_one_sample, 1)

        """

        # ---- Init the feature & Label ---- #
        # weight, shape=(sample_num, tick_num_in_one_sample, LS, 1)
        weight = np.zeros([self.date_num, self.tick_num_in_one_sample, self.LS, 1])
        # label, shape=(sample_num, tick_num_in_one_sample, LS, 1)
        label = np.zeros([self.date_num, self.tick_num_in_one_sample, self.LS, 1])
        # feature dict, each item shape=(sample_num, tick_num_in_one_sample, S, feature_shape)
        feature_dict = {}
        for feature_type in self.features_shape_dict.keys():
            feature_dict[feature_type] = np.zeros((self.date_num, self.tick_num_in_one_sample, self.S) + self.features_shape_dict[feature_type][1:])

        # ---- Load Labels and Features ---- #
        # grab `sample_num` samples of one tick
        for tick_n in range(self.tick_num_in_one_sample):
            # compute the target tick idx = start_idx + shift_idx + tick_gap
            tick_idx = self.start_tick - 1 + idx + tick_n * self.tick_gap
            assert tick_idx <= self.end_tick - 1, "Tick idx is out of boundary !"
            # read the label based on the tick_idx
            label[:, tick_n] = self.stack_label_array[:, tick_idx - self.LS + 1:tick_idx + 1]
            # read the weight based on the tick_idx
            weight[:, tick_n] = self.stack_weight_array[:, tick_idx - self.LS + 1:tick_idx + 1]
            # read the feature based on the tick
            for feature_type, data in feature_dict.items():
                if feature_type not in self.paf_shift_k_dict.keys():
                    data[:, tick_n] = self.stack_features_array_dict[feature_type][:, tick_idx - self.S + 1:tick_idx + 1]
                else:  # PAF features, do lag and read by `hf`
                    hf = self.paf_hf_dict[feature_type]  # read the hf
                    paf_shift_k = self.paf_shift_k_dict[feature_type]  # read the shift_k
                    data[:, tick_n] = self.stack_features_array_dict[feature_type][:, tick_idx - self.S + 1:tick_idx + 1, (tick_idx + 1) % hf]
                    if paf_shift_k > 0:  # shift the paf k, if paf_shift_k > 0
                        data[:, tick_n, :self.S - paf_shift_k] = \
                            self.stack_features_array_dict[feature_type][:, tick_idx - self.S + 1:tick_idx - paf_shift_k + 1, (tick_idx - paf_shift_k + 1) % hf]

        # ---- Reshape All Features, Label and Weight ---- #
        # - For feature, shape=(sample_num * tick_num_in_one_sample, 1, shape)
        for feature_type, data in feature_dict.items():
            feature_dict[feature_type] = data.reshape((self.date_num * self.tick_num_in_one_sample, 1, self.S) + self.features_shape_dict[feature_type][1:])
        # - For label, shape=(sample_num * tick_num_in_one_sample, LS, 1)
        label = label.reshape((self.date_num * self.tick_num_in_one_sample, self.LS, 1))
        # - For weight, shape=(sample_num * tick_num_in_one_sample, LS, 1)
        weight = weight.reshape((self.date_num * self.tick_num_in_one_sample, self.LS, 1))

        # ---- Construct item data ---- #
        item_data = {"features": {}, "label": label, "weight": weight}
        for feature_type in feature_dict.keys():
            item_data["features"][feature_type] = feature_dict[feature_type]
        return item_data


class MultiCodesDailySample(DailySampleBase):
    """ Characterizes the daily sample dataset of deep-lob for PyTorch.
        This is the new type Dataset, get (days * tick_num_in_one_sample) tick item each time.
        Drop the meaningless data, set all weights be 1.

    """

    def __len__(self) -> int:
        """ The total number of tick items. All possible items index. """

        return self.tick_gap

    def __getitem__(self, idx) -> dict:
        """ Get the item based one idx, and lag the item.

        return: item_data, (one tick of one future)
            - `features`: shape=(sample_num*tick_num_in_one_sample, seq, code, *feature.shape)
            - `label`: the return label, shape=(sample_num*tick_num_in_one_sample, code, seq, 1)
            - `weight`: the weight, shape=(sample_num*tick_num_in_one_sample, code, seq, 1)

        """
        # ---- Init the feature & Label ---- #
        label = np.zeros([self.date_num, self.tick_num_in_one_sample, self.code_num, self.LS, 1])
        feature_dict = {}
        for feature_type in self.features_shape_dict.keys():
            feature_dict[feature_type] = np.zeros((self.date_num, self.tick_num_in_one_sample,
                                                   self.feature_code_num, self.S,) + self.features_shape_dict[feature_type][2:])
        weight = np.zeros_like(label)
        # ---- Load Labels and Features ---- #
        # grab `sample_num` samples of one tick
        for tick_n in range(self.tick_num_in_one_sample):
            # compute the target tick idx = start_idx + shift_idx + tick_gap
            tick_idx = self.start_tick - 1 + idx + tick_n * self.tick_gap
            assert tick_idx <= self.end_tick - 1, "Tick idx is out of boundary !"
            # read the label based on the tick_idx
            label[:, tick_n] = self.stack_label_array[:, :, tick_idx - self.LS + 1:tick_idx + 1]
            # read the weight based on the tick_idx
            weight[:, tick_n] = self.stack_weight_array[:, :, tick_idx - self.LS + 1:tick_idx + 1]
            # read the feature based on the tick
            for feature_type, data in feature_dict.items():
                if feature_type not in self.paf_shift_k_dict.keys():
                    data[:, tick_n] = self.stack_features_array_dict[feature_type][:, :, tick_idx - self.S + 1:tick_idx + 1]
                else:  # PAF features, do lag and read by `hf`
                    hf = self.paf_hf_dict[feature_type]  # read the hf
                    paf_shift_k = self.paf_shift_k_dict[feature_type]  # read the shift_k
                    data[:, tick_n] = self.stack_features_array_dict[feature_type][:, :, tick_idx - self.S + 1:tick_idx + 1, (tick_idx + 1) % hf]
                    if paf_shift_k > 0:  # shift the paf k, if paf_shift_k > 0
                        data[:, tick_n, :self.S - paf_shift_k] = \
                            self.stack_features_array_dict[feature_type][:, tick_idx - self.S + 1:tick_idx - paf_shift_k + 1, (tick_idx - paf_shift_k + 1) % hf]

        # ---- Reshape All Features, Label and Weight ---- #
        # - For feature, shape=(sample_num * tick_num_in_one_sample, code_num, shape)
        for feature_type, data in feature_dict.items():
            feature_dict[feature_type] = data.reshape((self.date_num * self.tick_num_in_one_sample,) + (self.feature_code_num, self.S,)
                                                      + self.features_shape_dict[feature_type][2:])
        # - For label, shape=(sample_num * tick_num_in_one_sample, LS, 1)
        label = label.reshape((self.date_num * self.tick_num_in_one_sample, self.code_num, self.LS, 1))
        # - For weight, shape=(sample_num * tick_num_in_one_sample, LS, 1)
        weight = weight.reshape((self.date_num * self.tick_num_in_one_sample, self.code_num, self.LS, 1))

        # ---- Construct item data ---- #
        item_data = {"features": {}, "label": label, "weight": weight}
        for feature_type in feature_dict.keys():
            item_data["features"][feature_type] = feature_dict[feature_type]
        return item_data
