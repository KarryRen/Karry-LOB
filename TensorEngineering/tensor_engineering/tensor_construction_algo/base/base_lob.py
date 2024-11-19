# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/28 11:25
#
# pylint: disable=no-member

""" The LOB algorithm. """

import numpy as np

from ..tensor_construction_base import ConstructionAlgoBase
from Ops import Code


class LOB(ConstructionAlgoBase):
    """ Get the Limit Order Book data. """

    def __init__(self, **kwargs):
        # ---- Define the special params for LOB ---- #
        version = (2024, 2, 22, 0, 0)  # use date to note version
        out_coords = {
            "T": list(range(28800)),
            "D": ["Bid", "Ask"],
            "L": list(range(1, 6)),
            "F": ["Price", "Volume"]
        }  # the out coords
        data_source = {"LOB": {"features": ["Price", "Volume"], "levels": 5}}  # the data source

        # ---- Init the base Class ---- #
        super(LOB, self).__init__(version=version, out_coords=out_coords, data_source=data_source, **kwargs)

    def cal_fea(self, code: Code, date: str):
        """ LOB calculation algorithm. """

        # ---- Step 1. Get the market data of the `code` in `date` based on the data_source ---- #
        # ds_array_well is the flag of reading data_source right or not
        # ds_xray_dict is the dict of each data_source type and xarray
        ds_array_well, ds_xray_dict = self.get_all_market_data(code, date)

        # ---- Step 2. Get the LOB based on the ds_xray ---- #
        if np.all(ds_array_well):  # all datasource array well
            # Read the data
            xray_lob = ds_xray_dict["LOB"]
            # Set the LOB data
            self.xray.data[:] = xray_lob.data
            # Make price be int
            self.xray.loc[:, :, :, "Price"] *= 100
        return self.xray
