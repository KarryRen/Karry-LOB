# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/22 16:22

""" The interfaces of deeplob dataset. Now only support 1 code! """

from typing import Union

from .tick_sample import TickSampleDataset
from .daily_sample import DailySampleDataset
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
            return DailySampleDataset
        else:
            return TickSampleDataset
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
