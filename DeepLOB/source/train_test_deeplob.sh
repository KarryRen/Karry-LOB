#!/bin/bash

# @Author  : Karry Ren
# @Time    : 2024/11/21 16:51
# @Comment : This is the entrance for training and testing deeplob.
#            Don't run this directly, but use the `run_deeplob.sh`

# ---- Some environment settings ---- #
# shellcheck disable=SC2155
export PYTHONPATH=$(pwd):$PYTHONPATH
export redo="0" # redo or not

# ---- Set the config version ---- #
export config_version="0.0.1.OnlyLOB.0"
echo "** You are running the config version: '$config_version' NOW."

# ---- Some OS environment params ---- #
export test_start_month="202209" # the start test month, format must be "yyyymm"
export test_end_month="202301" # the start test month, format must be "yyyymm"
export test_freq="W" # the freq of rolling test period (we can do this by monthly 'M' too)
export test_step=2 # the step of rolling test period, based on test_freq
export train_valid_num=20 # the num of train and valid weeks, based on test_freq
export valid_date_num=20 # the num of valid dates
export task_config_path="DeepLOB/task_config.yaml" # the config `.json` file for tasks

# ---- Running the python script ---- #
python3.8 DeepLOB/launch/train_test_deeplob.py
