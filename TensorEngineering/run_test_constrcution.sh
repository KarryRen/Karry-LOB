#!/bin/bash

# @author  : Karry Ren
# @time    : 2024/11/20 10:18
# @comment : This is the entrance for Testing Construction of Tensor Engineering.
#            Run this by `./run_test_construction.sh` directly
#            Please run this before you do the construction.

# ---- Some environment settings ---- #
# shellcheck disable=SC2155
export PYTHONPATH=$(pwd):$PYTHONPATH

# ---- Run test the tensor construction algorithm ---- #
python3.8 TensorEngineering/test/test_construction.py
