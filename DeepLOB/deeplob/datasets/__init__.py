# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 3/15/24 5:30 PM
#
# pylint: disable=no-member

""" The interfaces of deeplob dataset. Total having 4 kinds ! """

from typing import Union

from .tick_sample import DeepLOBDataset, MultiCodesDeepLOB
from .daily_sample import DeepLOBDataset_DailySample, MultiCodesDailySample
from .collate import DailySampleCollate, TickSampleCollate


def get_class_of_dataset(class_type: str = None, codes: Union[list, str] = None):
    """ Get the class of deeplob dataset.

    :param class_type: the type of dataset class, you have only 2 choices now:
        - `DeepLOBDataset_DailySample`
        - `DeepLOBDataset`
    :param codes: the codes to model, single code or multi codes

    """

    if isinstance(codes, str):  # for single code
        if class_type == "DeepLOBDataset_DailySample":
            return DeepLOBDataset_DailySample
        else:
            return DeepLOBDataset
    elif isinstance(codes, list):  # for multi codes
        if class_type == "DeepLOBDataset_DailySample":
            return MultiCodesDailySample
        else:
            return MultiCodesDeepLOB
    else:
        raise TypeError(class_type)


def get_class_of_collate(class_type: str = None):
    """ Get the class of deeplob dataset collate.

    :param class_type: the type of dataset class, you have only 2 choices now:
        - `DeepLOBDataset_DailySample`
        - `DeepLOBDataset`

    """

    if class_type == "DeepLOBDataset_DailySample":
        return DailySampleCollate
    else:
        return TickSampleCollate
