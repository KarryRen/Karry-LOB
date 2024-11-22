# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2023/12/14 10:25

""" Parsing the `task_id` and building the associated `environ_config` that matches
    it to ensure that the `config` files can do the correct setting control.

"""

import os
import json
import yaml
import h5py
from typing import List, Union
from itertools import product
import numpy as np
import pandas as pd

from Ops.util import get_trading_dates
from Ops import CodeType, Date


def valid_for_one_date(date: Date, codes: Union[str, List[str]]) -> bool:
    """ Valid the data for one date of all codes.

    :param date: the date to valid
    :param codes: the code

    :return: bool_valid, valid or not
    """

    # ---- Transfer str codes to List[str] ---- #
    if isinstance(codes, str):
        codes = [codes]

    # ---- Valid the date base on the HFTLabel ---- #
    bool_valid = True
    for code in codes:
        valid_data_path = f"/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering/{code}/{date}/HFTLabel.h5"
        if os.path.exists(valid_data_path):  # file is existed but max value is 0 -> not valid
            with h5py.File(valid_data_path, "r") as fp:
                max_value = np.max(np.abs(fp["__xarray_dataarray_variable__"][:]))
            if max_value <= 1e-8:  # max data is not enough
                bool_valid = False
        else:  # file is not existed -> not valid
            bool_valid = False
        if not bool_valid:  # for one code is not valid then is not valid
            break
    return bool_valid


def get_dates_for_valid_market_data(dates: List[Date], codes: Union[str, List[str]]) -> List[Date]:
    """ Get the dates for valid market data.

    :param dates: the raw market data
    :param codes: the code list or just one code

    return:
        - valid_dates: the list of valid dates for the codes

    """

    # ---- Set the [start_date, end_date) of valid interval and select the date ---- #
    valid_start_date, valid_end_date = Date("20220101"), Date("20230101")
    dates = [date for date in dates if valid_start_date <= date < valid_end_date]

    # ---- Delete the wrong date for codes ---- #
    valid_dates = []
    have_met_first_valid = False  # meet first valid date or not
    for date in dates:
        bool_valid = valid_for_one_date(date, codes)
        if bool_valid:
            have_met_first_valid = True
            valid_dates.append(date)
        else:
            if have_met_first_valid:  # only print those missing dates after the first valid date (avoid too much print)
                print(f"!! SKIP: skip a not valid date: {date}")
    return valid_dates


# ---- Get `all` trading dates list ["yyyymmdd_1", "yyyymmdd_2", ..., "yyyymmdd_n"]---- #
market_trading_date_list = get_trading_dates()

# ---- Initialize the environment config ---- #
environ_config = {
    "config_version": "None",  # the version of config, "None" or "x.y.z"
    "redo": int(1),  # 0 for incremental train/test
    "job_id": "None",  # the id of one job
    "task_id": int(0),  # the id of task in one job
    "config_id": int(0),  # id for task config (the line in `task_config_path`)
    "period_id": int(0),  # id of period, used for cut the dates
    "task_config_path": "",  # test config file path
    "test_start_month": "None",  # test start month
    "test_end_month": "None",  # test end month
    "test_freq": "W",  # the frequency of rolling test period
    "test_step": int(1),  # the step of rolling test period
    "train_valid_num": int(5),  # the num of train & valid month
    "valid_date_num": int(20),  # the num of valid dates
    "future_type": "str",  # the types of future
    "seed_model": int(0),  # the rand seed for model
    "seed_dataloader": int(0),  # the rand seed for dataloader
    "seed_train": int(0),  # the rand seed for train

}

# ---- Get the os environs and set them to environ_config ---- #
for key in environ_config.keys():
    if os.environ.get(key) is not None:
        if isinstance(environ_config[key], list):  # do the `list` type transform
            environ_config[key] = type(environ_config[key])(eval(os.environ.get(key)))
        else:  # do the type transform
            environ_config[key] = type(environ_config[key])(os.environ.get(key))

# ---- Transfer the task_id to month_id and config_id based on the `task_config_file` ---- #
# read the task_config_file to get the total task num
task_config_path = environ_config["task_config_path"]
assert os.path.exists(task_config_path), f"The `task_config_path` you set `{task_config_path}` is not existed !! "
with open(task_config_path, "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
task_key = list(config_yaml.keys())
task_values_list = list(product(*config_yaml.values()))  # `product` means Cartesian product
# get the total number of task config
all_task_config_num = len(task_values_list)
# the `config_id` controls the lots of settings, such as future_type, random_seed ... in Step 5.
environ_config["config_id"] = environ_config["task_id"] % all_task_config_num
# the `period_id` controls the splitting of dates
environ_config["period_id"] = environ_config["task_id"] // all_task_config_num

# ---- Update environ_config based on the `config_id` ---- #
# get the task_values, a tuple (future_type, seed_model, seed_train, ...)
task_values = task_values_list[environ_config["config_id"]]
# for loop the task_key and set the value to environ_config
for k, key in enumerate(task_key):
    assert key in environ_config.keys(), f"The `{key}` in `{task_config_path}` is not valid !!"
    default_value = environ_config[key]  # get the default value of environ_config
    if isinstance(default_value, list):  # do the `list` type transform
        environ_config[key] = eval(task_values[k])
    elif default_value:  # do the other type transform
        environ_config[key] = type(default_value)(task_values[k])
    else:  # default value is None
        environ_config[key] = task_values[k]

# ---- Split the train, valid and test freq_bar dates ---- #
# get the valid date for `environ_config["future_type"]`
market_trading_date_list = get_dates_for_valid_market_data(market_trading_date_list, environ_config["future_type"])
# get the valid market trading month (yyyymm) list, use set to guarantee never repeated !!
market_trading_month_list = sorted(set([x.format_month() for x in market_trading_date_list]))
# select all test months in [test_start_month, test_end_month)
all_test_month_list = [x for x in market_trading_month_list if environ_config["test_start_month"] <= x < environ_config["test_end_month"]]
# construct the df for trading date
market_trading_date_df = pd.DataFrame(market_trading_date_list, columns=["date"])
# generate the freq_key based on the date
if environ_config["test_freq"] == "M":  # Monthly
    # compute the corresponding `month` for each date, yyyymm
    market_trading_date_df.loc[:, "freq_key"] = [x.strftime("%Y%m") for x in market_trading_date_df["date"]]
elif environ_config["test_freq"] == "W":  # Weekly
    # compute the corresponding `week` for each date, yyyyWeekNum, such as 199051 meaning the 51^th week in year 1990
    market_trading_date_df.loc[:, "freq_key"] = [x.strftime("%Y%W") for x in market_trading_date_df["date"]]
elif environ_config["test_freq"] == "D":  # Daily
    # compute the corresponding `day` for each date, yyyymmdd
    market_trading_date_df.loc[:, "freq_key"] = [x.strftime("%Y%m%d") for x in market_trading_date_df["date"]]
else:
    raise ValueError(environ_config["test_freq"])
# compute the `begin` and `end` date of each freq
temp_freq_df = market_trading_date_df.groupby(market_trading_date_df["freq_key"]).last()
temp_freq_df.loc[:, "begin_date"] = market_trading_date_df.groupby(market_trading_date_df["freq_key"]).first()
temp_freq_df.loc[:, "end_date"] = market_trading_date_df.groupby(market_trading_date_df["freq_key"]).last()
# the freq_bar is a list of tuple, each tuple means a freq period:
# [(yyyymmdd_of_freq_begin, yyyymmdd_of_freq_end), (yyyymmdd_of_freq_begin, yyyymmdd_of_freq_end), ..., ]
freq_bar = [(Date(x[0].astype(str)), Date(x[1].astype(str))) for x in temp_freq_df[["begin_date", "end_date"]].values]
# collect all test and train&valid tuple by freq_bar, one for one
# get the list of test_freq_bar_dates (list of tuple)
# get the list of train&valid_freq_bar_dates (list of List[tuple])
all_test_freq_bar_dates, all_train_valid_freq_bar_dates = [], []
for k in range(1, len(freq_bar)):
    if freq_bar[k][0].format_month() in all_test_month_list or freq_bar[k][1].format_month() in all_test_month_list:
        all_test_freq_bar_dates.append(freq_bar[k])  # append 1 freq_bar for test
        all_train_valid_freq_bar_dates.append(
            freq_bar[k - environ_config["train_valid_num"]:k])  # append `train_valid_num` freq_bar (list) for train&valid
# get the task-based dates based on the test_step and period_id
task_test_freq_bar_dates = all_test_freq_bar_dates[environ_config["test_step"] * environ_config["period_id"]:
                                                   environ_config["test_step"] * (1 + environ_config["period_id"])]
task_train_valid_freq_bar_dates = all_train_valid_freq_bar_dates[environ_config["test_step"] * environ_config["period_id"]:
                                                                 environ_config["test_step"] * (1 + environ_config["period_id"])]

# ---- Get the final train, valid and test dates ---- #
test_dates, train_valid_dates = [], []
for freq_bar_dates in task_test_freq_bar_dates:  # get all final test dates from all tuple
    test_dates.extend([x for x in market_trading_date_list if freq_bar_dates[0] <= x <= freq_bar_dates[1]])
for freq_bar_dates in task_train_valid_freq_bar_dates[0]:  # get all final train&valid dates from the first(earliest) tuple list
    train_valid_dates.extend([x for x in market_trading_date_list if freq_bar_dates[0] <= x <= freq_bar_dates[1]])
environ_config["train_dates"] = [str(x) for x in train_valid_dates[:-environ_config["valid_date_num"]]]
environ_config["valid_dates"] = [str(x) for x in train_valid_dates[-environ_config["valid_date_num"]:]]
environ_config["test_dates"] = [str(x) for x in test_dates]

# ----- Print all task info ---- #
print(f"***************** HERE IS TASK INFO ! *****************")
print(json.dumps(environ_config.__str__(), indent=4))
print(f"******************* TASK INFO END ! *******************\n")
