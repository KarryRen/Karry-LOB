# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/04/11 11:25

""" Some util functions. """

import h5py
import os
import datetime
from typing import Callable, List, Union
import numpy as np
import akshare as ak

from .datatype import Code, CodeType, Exchange, Date

global_switch_hour = 18


def get_trade_date_list(month, lookback_days=(0, 0)):
    """
    获取当前数据集支持的日期列表.
    lookback_days为二元组。分别表示历史和未来的扩张天数
    """
    this_hour = int(datetime.datetime.now().strftime('%H'))
    if os.environ.get("realtime", "0") == "1":
        end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y%m%d')
    else:
        if this_hour < global_switch_hour:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
        else:
            end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y%m%d')
    date_list = [x.replace('-', '') for x in TradingDays()]
    if lookback_days != (0, 0):
        start_index = date_list.index([x for x in date_list if x.startswith(month)][0]) - lookback_days[0]
        end_index = date_list.index([x for x in date_list if x.startswith(month) and x < end_date][-1]) + 1 + lookback_days[1]
        date_list = date_list[start_index:end_index]
    else:
        date_list = [x for x in date_list if x.startswith(month) and x < end_date]
    date_list = [Date(x) for x in date_list]
    return date_list


def get_trading_dates(
        start_date: Union[datetime.datetime, bytes, str] = None,
        end_date: Union[datetime.datetime, bytes, str] = None,
        month: str = None
) -> List[Date]:
    """ Get the trading dates.

    :param start_date: the start date, if str format should be `yyyy-mm-dd` or `yyyymmdd`
    :param end_date: the end date, if str format should be `yyyy-mm-dd` or `yyyymmdd`
    :param month: the month, format should be `yyyymm`

    :return: dates - the list of all filtered dates

    Attention:
        - `date interval` is [start_date, end_date), front close and end open
        - `month` is used to do the start with selection

    """

    # ---- Get all trading days and transfer them to Date() Class ---- #
    dates = [Date(x.strftime("%Y%m%d")) for x in ak.tool_trade_date_hist_sina()["trade_date"]]

    # ---- Filter the dates with [star_date, end_date) or start_with(month) ---- #
    if start_date is not None:
        start_date = Date(start_date)
        dates = [x for x in dates if x >= start_date]
    if end_date is not None:
        end_date = Date(end_date)
        dates = [x for x in dates if x < end_date]
    if month is not None:
        dates = [x for x in dates if str(x).startswith(month)]

    # ---- Return the dates ---- #
    return dates

def get_real_code_by_symbol(symbols: Code, date):
    """

    """
    base_dir = f"/home/fisher_research/Data/50MS_Md"
    if os.access(base_dir, os.R_OK):
        real_code_dict = {}
        for symbol in symbols:
            path = f"{base_dir}/2019Data_{symbol}/{date[:6]}.h5"
            if os.path.exists(path):
                with h5py.File(path, "r") as fp:
                    real_code_dict[symbol] = Code(fp['InstrumentId'][date][0].astype(np.str_))
    else:
        raise NotImplementedError()
    return real_code_dict
