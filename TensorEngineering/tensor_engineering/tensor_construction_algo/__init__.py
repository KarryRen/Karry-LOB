# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/02/28 10:25

""" The interface of Tensor Construction Algorithm. """

import os

# ---- The base algo class ---- #
from .tensor_construction_base import ConstructionAlgoBase

# ---- Import all possible Algo Class ---- #

# ---- Import the base algo class if `tensor_group` is `base` ---- #
if os.environ.get("tensor_group") == "base":
    from .base import *
