# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The interfaces of module. """

from .base import VoidModule

# ---- Interfaces of norm ---- #
from .norm import get_norm_instance

# ---- Interfaces of conv ---- #
from .conv import get_conv2d_instance
