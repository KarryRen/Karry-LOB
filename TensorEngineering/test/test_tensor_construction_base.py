# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/27 10:25
#
# pylint: disable=no-member

""" Testing the basic feature algorithms. """

from TensorEngineering.TensorConstructionAlgo.tensor_construction_base import ConstructionAlgoBase

base = ConstructionAlgoBase()
base.data_source = {
    "LOB": {},
    "TradeVolume": {}
}
_bool, xarray = base.get_all_market_data("IC_M0", "20231010")
print(_bool)
print(xarray)
base.check_data("IC_M0", "20231010", xarray)
