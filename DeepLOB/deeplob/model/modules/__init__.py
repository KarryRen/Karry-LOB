# -*- coding: utf-8 -*-
# @author : MaMing, RenKai (intern in HIGGS ASSET)
# @time   : 3/23/24 1:05 PM
#
# pylint: disable=no-member

""" The interfaces of module. """

from .base import VoidModule

# ---- Interfaces of norm ---- #
from .norm import get_norm_instance

# ---- Interfaces of conv ---- #
from .conv import get_conv2d_instance
