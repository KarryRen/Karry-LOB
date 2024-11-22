# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 4/8/24 5:30 PM
#
# pylint: disable=no-member

""" DeepLOBDataset(default is tick sample) for deep-lob net.

"""

import numpy as np
from typing import Tuple

from .base import TickSampleBase


class DeepLOBDataset(TickSampleBase):
    """ Characterizes the dataset of deep-lob for PyTorch.
        This is the traditional type Dataset, only get (one) tick item each time.
        Keep the meaningless data, set the weights of meaningful data be 1, meaningless data be 0.

    """

    def __len__(self) -> int:
        """ The total number of tick items. All possible items index. """

        return self.date_num * self.tick_num

    def __getitem__(self, idx) -> Tuple:
        """ Get the item based on idx, and lag the item.

        return: item_data (one tick of one future)
            - `features`: all needed features, shapes are different (corr to feature type).
            - `label`: the return label, shape=(label_len, 1)
            - `weight`: the weight, shape=(label_len, 1)
            - `metadata`: the metadata, dict

        """

        # ---- Compute the index pair [date_idx, tick_idx] to locate data ---- #
        date_idx = idx // self.tick_num  # get the date index to locate the date of future
        tick_idx = idx % self.tick_num  # get the tick index to locate the tick of daily data

        # ---- Get the needed_features, label, w ---- #
        # feature dict, each item shape=(S, feature_shape)
        feature_dict = {}
        # meaningless data, features are made to all zeros
        if tick_idx < self.start_tick - 1 or tick_idx > self.end_tick - 1:
            # set features, all zeros, shape is different from feature to feature
            for feature_type in self.features_shape_dict.keys():
                feature_dict[feature_type] = np.zeros((self.S,) + self.features_shape_dict[feature_type][1:])
            # `label = 0.0` for loss computation, shape=(LS, 1)
            label = np.zeros((self.LS, 1))
            # `weight = 0.0` means data is meaningless, shape=(LS, 1)
            weight = np.zeros_like(label)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on feature type
            for feature_type in self.features_shape_dict.keys():
                if feature_type not in self.paf_shift_k_dict.keys():
                    feature_dict[feature_type] = self.features_list_dict[feature_type][date_idx][tick_idx - self.S + 1:tick_idx + 1]
                else:  # PAF, do lag and read by `hf`
                    hf = self.paf_hf_dict[feature_type]  # read the hf
                    paf_shift_k = self.paf_shift_k_dict[feature_type]  # read the shift_k
                    feature_dict[feature_type] = self.features_list_dict[feature_type][date_idx][tick_idx - self.S + 1:tick_idx + 1, (tick_idx + 1) % hf]
                    if paf_shift_k > 0:  # shift the paf K, if paf_shift_k > 0
                        feature_dict[feature_type][:self.S - paf_shift_k] = \
                            self.features_list_dict[feature_type][date_idx][tick_idx - self.S + 1:tick_idx - paf_shift_k + 1, (tick_idx - paf_shift_k + 1) % hf]
            # get the label, shape=(LS, 1)
            label = self.stack_label_array[date_idx][tick_idx - self.LS + 1:tick_idx + 1]
            # set the weight, shape=(LS, 1)
            weight = self.stack_weight_array[date_idx][tick_idx - self.LS + 1:tick_idx + 1]

        # ---- Construct item data ---- #
        item_data = {"features": {}, "label": label, "weight": weight, "metadata": {"date": np.array([self.dates[date_idx]])}}
        for feature_type, data in feature_dict.items():  # set features
            item_data["features"][feature_type] = np.array([data])
        return item_data


class MultiCodesDeepLOB(TickSampleBase):
    """ Characterizes the dataset of deep-lob for PyTorch.
        This is the traditional type Dataset, only get (one) tick item each time.
        Keep the meaningless data, set the weights of meaningful data be 1, meaningless data be 0.

    """

    def __len__(self) -> int:
        """ The total number of tick items. All possible items index. """

        return self.date_num * self.tick_num

    def __getitem__(self, idx) -> dict:
        """ Get the item based on idx, and lag the item.

        return: item_data (one tick of one future)
            - `features`: all needed features, shapes are different (corr to feature type).
            - `label`: the return label, shape=(label_len, 1)
            - `weight`: the weight, shape=(label_len, 1)
            - `metadata`: the metadata, dict

        """

        # ---- Compute the index pair [date_idx, tick_idx] to locate data ---- #
        date_idx = idx // self.tick_num  # get the date index to locate the date of future
        tick_idx = idx % self.tick_num  # get the tick index to locate the tick of daily data

        # ---- Get the needed_features, label, w ---- #
        # ---- Construct item data ---- #
        features_dict = {}
        # meaningless data, features are made to all zeros
        if tick_idx < self.start_tick - 1 or tick_idx > self.end_tick - 1:
            # set features, all zeros, shape is different from feature to feature
            for feature_type in self.features_shape_dict.keys():
                features_dict[feature_type] = np.zeros((self.feature_code_num, self.S,) + self.features_shape_dict[feature_type][2:])
            label = np.zeros((self.code_num, self.LS, 1))
            weight = np.zeros_like(label)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on feature type
            for feature_type in self.features_shape_dict.keys():
                if feature_type not in self.paf_shift_k_dict.keys():
                    features_dict[feature_type] = self.features_list_dict[feature_type][date_idx][:, tick_idx - self.S + 1:tick_idx + 1]
                else:  # PAF, do lag and read by `hf`
                    hf = self.paf_hf_dict[feature_type]  # read the hf
                    paf_shift_k = self.paf_shift_k_dict[feature_type]  # read the shift_k
                    features_dict[feature_type] = self.features_list_dict[feature_type][date_idx][:,
                                                  tick_idx - self.S + 1:tick_idx + 1, (tick_idx + 1) % hf]
                    if paf_shift_k > 0:  # shift the paf K, if paf_shift_k > 0
                        features_dict[feature_type][:self.S - paf_shift_k] = self.features_list_dict[feature_type][date_idx][
                                                                             :, tick_idx - self.S + 1:tick_idx - paf_shift_k + 1,
                                                                             (tick_idx - paf_shift_k + 1) % hf]
            label = self.stack_label_array[date_idx][:, tick_idx - self.LS + 1:tick_idx + 1]
            weight = self.stack_weight_array[date_idx][:, tick_idx - self.LS + 1:tick_idx + 1]
        # will del soon
        item_data = {"features": features_dict, "label": label, "weight": weight, "metadata": {"date": np.array([self.dates[date_idx]])}}
        return item_data
