# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/27 10:25
#
# pylint: disable=no-member

""" The core of testing feature construction algorithms. """

import numpy as np
from TensorEngineering.Ops import Code
from TensorEngineering.TensorEngineering.TensorConstructionAlgo.base.base_label import HFTLabel

# ---- Step 1. Define some `special` params for feature construction ---- #
code = Code("IF_M0", code_type="FINANCIAL_FUTURE")
date = "20230621"

# ---- Step 2. Define the CLASS of feature construction algorithm ---- #
algo = HFTLabel(code.code_type)

# ---- Step 3. Call the algorithm to compute feature ---- #
xray = algo(code, date, save=False)

# ---- Step 4. Print the detail information of the feature ---- #
print(xray)
print(np.mean(xray.data, axis=(0,)))
print(np.std(xray.data, axis=(0,)))
