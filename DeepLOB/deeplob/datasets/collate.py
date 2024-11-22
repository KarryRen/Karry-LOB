# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The collate class of dataset.

:TODO: remove dtype transfer in __call__() function.

"""

import torch
import numpy as np
from typing import List


class TickSampleCollate:
    """ The collate_fn() class of tick sample dataset. """

    def __init__(self, device: torch.device):
        """ Init of TickSampleCollate. """

        self.device = device

    def __call__(self, batch_data: List[dict]):
        """ collate_fn """

        # define item_data
        item_data = {}

        # creat keys for item_data, while change datatype and device
        for key in batch_data[0].keys():
            if isinstance(batch_data[0][key], dict):
                item_data[key] = {}
                for key2 in batch_data[0][key].keys():
                    if isinstance(batch_data[0][key][key2], dict):
                        raise ValueError(batch_data[0][key][key2])
                    else:
                        item_data[key][key2] = torch.Tensor(np.stack([sample[key][key2] for sample in batch_data])).to(device=self.device,
                                                                                                                       dtype=torch.float32)
            else:
                item_data[key] = torch.Tensor(np.stack([sample[key] for sample in batch_data])).to(device=self.device, dtype=torch.float32)

        # insert metadata into features
        item_data["features"]["metadata"] = item_data["metadata"]
        return item_data["features"], {"label": item_data["label"], "weight": item_data["weight"]}


class DailySampleCollate:
    """ The collate_fn() class of daily sample dataset. """

    def __init__(self, device: torch.device):
        """ Init of DailySampleCollate. """

        self.device = device

    def __call__(self, batch_data: List[dict]):
        """ collate_fn """

        # ---- Step 1. Construct the item data --- #
        item_data = {"features": {}, "label": [], "weight": []}

        # ---- Step 2. For loop to collect the batch_data ---- #
        for b_d in batch_data:
            # collect label
            item_data["label"].append(b_d["label"])
            # collect weight
            item_data["weight"].append(b_d["weight"])
            # collect features
            for feature_type in b_d["features"].keys():
                if feature_type not in item_data["features"].keys():
                    item_data["features"][feature_type] = []
                item_data["features"][feature_type].append(b_d["features"][feature_type])

        # ---- Step 3. Concat batch_data and change datatype&device ---- #
        # concat label
        item_data["label"] = torch.tensor(np.concatenate(item_data["label"], axis=0)).to(device=self.device, dtype=torch.float32)
        # concat weight
        item_data["weight"] = torch.tensor(np.concatenate(item_data["weight"], axis=0)).to(device=self.device, dtype=torch.float32)
        # concat features
        for feature_type in item_data["features"].keys():
            item_data["features"][feature_type] = torch.tensor(np.concatenate(item_data["features"][feature_type], axis=0)).to(device=self.device,
                                                                                                                               dtype=torch.float32)
        return item_data["features"], {"label": item_data["label"], "weight": item_data["weight"]}
