# -*- coding: utf-8 -*-
# @Author  : Karry Ren
# @Time    : 2024/03/15 10:25

""" The test function for construction. """

import numpy as np
from Ops import Code
from TensorEngineering.tensor_engineering.tensor_construction_algo.base import LOB

# ---- Define some `special` params for feature construction ---- #
code = Code("IF_M0", code_type="FINANCIAL_FUTURE")
date = "20220104"

# ---- Define the CLASS of feature construction algorithm ---- #
algo = LOB()

# ---- Call the algorithm to compute feature ---- #
xray = algo(code, date, save=False)

# ---- Print the detail information of the feature ---- #
print(xray)
print(np.mean(xray.data, axis=(0,)))
print(np.std(xray.data, axis=(0,)))
