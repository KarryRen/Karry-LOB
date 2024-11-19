# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/27 11:25
#
# pylint: disable=no-member

""" Test the io functions. """

from TensorEngineering.io import load_xarray_from_h5

input = {
    "code": "IC_M0",
    "date": "20220107",
    "name": "VolumeImbalance",
}
xray = load_xarray_from_h5(**input)

print(xray)
