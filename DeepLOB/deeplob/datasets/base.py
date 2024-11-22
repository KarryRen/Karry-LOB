# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 4/8/24 5:30 PM
#
# pylint: disable=no-member

""" The base class of deeplob dataset. """

import logging
from collections import defaultdict
from typing import List, Dict
from torch.utils.data import Dataset as TorchDataset
import numpy as np


def cal_sample_std_weight(prior_label: np.ndarray, prior_weight: np.ndarray):
    """ Calculate a new weighted base on the std of different sample label,
        in order to reduce the weight of high std day.

    :param prior_label: the prior label of all samples, shape=(sample_num, T, 1)
    :param prior_weight: the prior weight of all samples (default is 1), shape=(sample_num, T, 1)

    return:
        - The operated weight, shape=(sample_num, T, 1)

    """

    # ---- Step 1. Compute sample std ---- #
    sample_std = (np.nansum((prior_label ** 2) * prior_weight, axis=(1, 2), keepdims=True) /
                  np.nansum(prior_weight, axis=(1, 2), keepdims=True)) ** 0.5  # shape=(sample_num, 1, 1)

    # ---- Step 2. Threshold the sample std ---- #
    sample_std_threshold = np.median(sample_std)
    sample_std[sample_std < sample_std_threshold] = sample_std_threshold  # std is lower than threshold

    # ---- Step 3. Use the sample_std to compute sample weight ---- #
    sample_std_weight = 1.0 / sample_std
    sample_std_weight = sample_std_weight / np.mean(sample_std_weight)
    print("sample std threshold: ", sample_std_threshold)
    print("sample weight: ", sample_std_weight.reshape(-1))
    return prior_weight * sample_std_weight


def name_2_type(name):
    """
    resolver group name to feature/label/weight
    """

    if name.startswith("Label"):
        name_type = "label"
    elif name.startswith("Weight"):
        name_type = "weight"
    elif name.startswith("DataArray"):
        name_type = "feature"
    else:
        raise ValueError(name)
    return name_type


class SampleBase(TorchDataset):
    """ Characterizes the dataset of deep-lob for PyTorch.

    This is the traditional type Dataset, only get (one) tick item each time, we call it as TickSample.

    During the __getitem__(): Keep the meaningless data, set the weights of meaningful data be 1, meaningless data be 0.

    """

    def __init__(
            self, data_root_dict: Dict[str, np.ndarray],
            dates: List[str] = None, feature_len: int = 5, label_len: int = 1,
            start_tick: int = 241, end_tick: int = 28560,
            use_sample_std_weight: bool = False, logger=None, **kwargs
    ) -> None:
        """ Initialization of the SampleBase. LOAD `label` & `needed_features` from the data_dict cache in memory.
            The key operations are:
                1. For-loop the `future_types`.
                2. For-loop the `dates` of one future.
                3. For-loop the `needed_features` and read ALL features&label then concat by TYPE.
            To be more flexible, we have used `dict` to store feature data several times,
            which may affect the readability of the code, but it does allow for a more UNIFORM situation.

        Parameters of the overall data configuration.
            :param data_root_dict: the root dict of data.
            :param dates: the SORTED list of dates, such as [`yyyymmdd1`, `yyyymmdd2`, ...]
                Dates could be used to split `train | valid | test` datasets.
                Make sure the dates are sorted, so that the dateset will be sorted !
            :param feature_len: the num of lag steps
                In tick `t` we will collect the feature from `t - feature_len + 1` to `t`.
            :param label_len: the num of label length
                ATTENTION: This param might be CONFUSED. In tick `t` we will collect the label from
                    `t - label_len + 1` to `t` rather than `t` to `t + label_len - 1`, which is totally
                    different from our target.
            :param start_tick: the start tick
            :param end_tick: the end tick
                There are some NOTES about the tick (VERY IMPORTANT):
                    - tick_num should be EQUAL to the tick_num when do feature_engineering !
                    - start_tick and end_tick means the MEANINGFUL tick period of one day.
                    - the tick period is [start_tick, end_tick], FRONT CLOSED and BACK CLOSED !
                    - start_tick and end_tick are both BEGIN from 1 NOT 0 !
                    - default is [241, 28560] which means drop the front 2 minutes and end 2 minutes data.
            :param logger: the logging controller

        Parameters of appended settings.
            :param use_sample_std_weight: use daily R2 weight or not

        """

        # ---- Check & Set the params ---- #
        assert label_len <= feature_len, "The label length must <= feature_len !"
        self.logger = logger if logger is not None else logging
        self.start_tick, self.end_tick = start_tick, end_tick

        self.merge_along_date(data_root_dict, dates=dates, use_sample_std_weight=use_sample_std_weight)
        self.infer_shape()

        # ---- Get the key dim of each feature ---- #
        self.S = feature_len  # the seq num of Features
        self.LS = label_len  # the seq num of Labels
        self.features_shape_dict = {}  # feature shape dict, key is feature_type, value is shape_tuple
        self.paf_hf_dict = {}  # paf hf dict, key is paf_type, value is hf
        self.paf_shift_k_dict = kwargs.get("paf_shift_k")  # paf shift k dict, key is paf_type, value is shift_k
        for feature_type in self.features_list_dict:
            feature_shape = self.features_list_dict[feature_type][0].shape  # extract the shape
            if feature_type not in self.paf_shift_k_dict.keys():
                self.features_shape_dict[feature_type] = feature_shape
            else:  # PAF should extract the `hf` dim
                self.features_shape_dict[feature_type] = feature_shape[:1] + feature_shape[2:]
                self.paf_hf_dict[feature_type] = feature_shape[1]
                assert feature_type in self.paf_shift_k_dict.keys(), f"Please declare the shift_k for {feature_type} !"

    def infer_shape(self, ):
        # tick_num: the num of ticks of one day for one future
        if len(self.stack_label_array.shape) == 3:
            # single code
            self.date_num, self.tick_num, _ = self.stack_label_array.shape
        elif len(self.stack_label_array.shape) == 4:
            # split feature_code_num and label_code_num
            self.date_num, self.code_num, self.tick_num, _ = self.stack_label_array.shape
            self.feature_code_num = self.stack_features_array_dict[list(self.stack_features_array_dict.keys())[0]].shape[1]
        else:
            raise NotImplementedError()
        assert self.end_tick <= self.tick_num, "The end_tick must <= tick_num !"

    def merge_along_date(self, hg_ds, **kwargs):
        self.stack_features_array_dict = {}
        self.features_list_dict = defaultdict(list)
        tmp = []
        for datadict, metadata in hg_ds:
            date = int(metadata[b'date'].decode())
            if kwargs.get("dates"):
                # 增加一个过滤条件。这个逻辑在hg中不会触发
                if str(date) not in kwargs.get("dates"):
                    continue
            tmp.append([date, datadict])
        tmp.sort(key=lambda x: x[0])
        self.dates = [date for date, datadict in tmp]
        label_list = [data for _, datadict in tmp for name, data in datadict.items() if name_2_type(name) == "label"]
        weight_list = [data for _, datadict in tmp for name, data in datadict.items() if name_2_type(name) == "weight"]
        if len(weight_list) == 0:
            weight_list = [np.ones_like(data) for _, datadict in tmp for name, data in datadict.items() if name_2_type(name) == "label"]
            print("no label weight. use ones_like")
        else:
            print("use label weight. ", np.mean(weight_list[0]), np.std(weight_list[0]))
        for date, datadict in tmp:
            for name, data in datadict.items():
                if name_2_type(name) == "feature":
                    self.features_list_dict[name].append(data)
        self.logger.info(f"self.dates in DeepLOBDataset_DailySample is {self.dates}")
        for k, v in self.features_list_dict.items():
            self.stack_features_array_dict[k] = np.stack(v, axis=0)
        self.stack_label_array = np.stack(label_list, axis=0)
        self.stack_weight_array = np.stack(weight_list, axis=0)
        # compute the sample std weight
        if kwargs.get("use_sample_std_weight"):
            self.stack_weight_array = cal_sample_std_weight(self.stack_label_array, self.stack_weight_array)

    def __len__(self) -> int:
        """ The total number of tick items. All possible items index. """
        raise NotImplementedError("The `__len__()` function must implemented in subclasses !!")

    def __getitem__(self, idx) -> dict:
        """ Get the item based on idx, and lag the item. """
        raise NotImplementedError("The `__getitem__()` function must implemented in subclasses !!")


class TickSampleBase(SampleBase):
    """ Characterizes the daily sample dataset of deep-lob for PyTorch.
        This is the new type Dataset, get (days * tick_num_in_one_sample) tick item each time.
        Drop the meaningless data, set all weights be 1.

    """

    def __init__(self, **kwargs) -> None:
        """ 
        Parameters of sub class
            :param tick_num_in_one_sample: tick_num_in_one_sample

        """
        super(TickSampleBase, self).__init__(**kwargs)


class DailySampleBase(SampleBase):
    """ Characterizes the daily sample dataset of deep-lob for PyTorch.
        This is the new type Dataset, get (days * tick_num_in_one_sample) tick item each time.
        Drop the meaningless data, set all weights be 1.

    """

    def __init__(self, **kwargs) -> None:
        """ 
        Parameters of sub class
            :param tick_num_in_one_sample: tick_num_in_one_sample

        """
        super(DailySampleBase, self).__init__(**kwargs)
        tick_num_in_one_sample = kwargs.get("tick_num_in_one_sample", 10)
        # ---- Daily Sample Config ---- #
        self.tick_num_in_one_sample = tick_num_in_one_sample
        total_tick_num_of_one_sample = self.end_tick - self.start_tick + 1
        assert total_tick_num_of_one_sample % tick_num_in_one_sample == 0, \
            (f"Total tick num `{total_tick_num_of_one_sample}` can't be divided by "
             f"one sample tick num `{tick_num_in_one_sample}`")
        self.tick_gap = total_tick_num_of_one_sample // tick_num_in_one_sample
