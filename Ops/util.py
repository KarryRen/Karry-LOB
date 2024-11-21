# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/04/11 11:25

""" Some util functions. """

import h5py
import os
import datetime
from typing import Callable, List, Union
import numpy as np
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


def get_dates(
        start_date: Union[datetime.datetime, bytes, str] = None, end_date: Union[datetime.datetime, bytes, str] = None,
        month: str = None
) -> List[Date]:
    """ Get the dates.

    :param start_date: the start date, if str format should be `yyyy-mm-dd` or `yyyymmdd`
    :param end_date: the end date, if str format should be `yyyy-mm-dd` or `yyyymmdd`
    :param month: the month, format should be `yyyymm`

    return: days-the list of all filtered dates

    Attention:
        - `date interval` is [start_date, end_date), front close and end open
        - `month` is used to do the start with selection

    """

    # ---- Get all trading days and transfer them to Date() Class ---- #
    dates = [Date(x) for x in TradingDays()]

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


def get_codes(date: datetime, code_type: Union[str, CodeType] = None, exchange: Exchange = "UNKNOWN", filter: Union[str, Callable] = None) -> List[
    Code]:
    """
    date: 
    code_type: 
    exchange: 
    filter: 
    """
    # ensure code_type is CodeType
    if isinstance(code_type, str):
        code_type = CodeType[code_type]
    # ensure exchange is exchange
    if isinstance(exchange, str):
        exchange = Exchange[exchange]
    codes = []
    # 根据code_type过滤
    if code_type == CodeType["FINANCIAL_FUTURE"]:
        codes = ["IH_M0", "IF_M0", "IC_M0", "IM_M0",
                 "IH_M1", "IF_M1", "IC_M1", "IM_M1"]
    elif code_type == CodeType["ETF"]:
        etf_sh = [
            "510050.SH",  # 华夏上证50ETF
            "510300.SH",  # 华泰柏瑞沪深300ETF
            "510500.SH",  # 南方中证500ETF
            "512100.SH",  # 南方中证1000ETF
            "563300.SH",  # 华泰柏瑞中证2000ETF
            "588000.SH",  # 华夏上证科创板50成份ETF
            "588080.SH",  # 易方达上证科创板50ETF
            "588030.SH",  # 博时上证科创板100ETF
            "588800.SH",  # 华夏上证科创板100ETF
        ]
        etf_sz = [
            "159901.SZ",  # 深证100
            "159915.SZ",  # 创业板ETF
            "159919.SZ",  # 沪深300ETF
            "159922.SZ",  # 中证500ETF
            "159845.SZ",  # 中证1000ETF
            "159531.SZ",  # 中证2000ETF
        ]
        codes = etf_sh + etf_sz
    elif code_type == CodeType["ETF_OPTION"]:
        etf_sh = [
            "510050.SH",  # 华夏上证50ETF
            "510300.SH",  # 华泰柏瑞沪深300ETF
            "510500.SH",  # 南方中证500ETF
            "588000.SH",  # 华夏上证科创板50成份ETF
            "588080.SH",  # 易方达上证科创板50ETF
        ]
        etf_sz = [
            "159901.SZ",  # 深证100
            "159915.SZ",  # 创业板ETF
            "159919.SZ",  # 沪深300ETF
            "159922.SZ",  # 中证500ETF
        ]
        codes = etf_sh + etf_sz
    elif code_type == CodeType["BOND"]:
        import pandas as pd
        daily_root_dir = "/mnt/weka/home/fisher_research/Data/HFT_Daily/BOND"
        last_date = sorted([x for x in os.listdir(daily_root_dir) if Date(x.split('.')[0]) <= date])[-1]
        df_path = os.path.join(daily_root_dir, last_date)
        bond_pool_df = pd.read_csv(df_path)
        # 过滤条件
        if filter:
            bond_pool_df = filter(bond_pool_df)
        codes = bond_pool_df[["债券代码", "交易所"]].apply(lambda x: '%d.%s' % (tuple(x.values)), axis=1).values
    else:
        raise ValueError(code_type)
    codes = [Code(x, code_type=code_type) for x in codes]

    # 根据exchange过滤
    codes = [code for code in codes if exchange == Exchange(0) or code.exchange == exchange]

    # 根据filter_func过滤
    if filter:
        if isinstance(filter, str):
            codes = [code for code in codes if filter in str(code)]
        else:
            codes = [code for code in codes if filter(code)]
    return codes


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
