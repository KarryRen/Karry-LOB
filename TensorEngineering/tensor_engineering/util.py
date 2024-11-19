# -*- coding: utf-8 -*-
# @Time    : 2024/11/19 18:06
# @Author  : Karry Ren

""" The utils functions for tensor engineering. """

import sys
import os
import datetime
from typing import Union, List
import akshare as ak

from Ops import Code, CodeType

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
    date_list = [x.strftime("%Y%m%d") for x in ak.tool_trade_date_hist_sina()["trade_date"]]
    if lookback_days != (0, 0):
        start_index = date_list.index([x for x in date_list if x.startswith(month)][0]) - lookback_days[0]
        end_index = date_list.index([x for x in date_list if x.startswith(month) and x < end_date][-1]) + 1 + lookback_days[1]
        date_list = date_list[start_index:end_index]
    else:
        date_list = [x for x in date_list if x.startswith(month) and x < end_date]
    return date_list


def gen_date_list(start: str, end: str) -> List[str]:
    """ Generate the trade date list in [start, end).

    :param start: start date
    :param end: end date

    :return: trade date list (list of `yyyymmdd`) from recent to before !

    """

    date_list = [x.strftime("%Y%m%d") for x in ak.tool_trade_date_hist_sina()["trade_date"]]
    date_list = [x for x in date_list if start <= x < end]
    date_list = date_list[::-1]
    return date_list


def get_code_list(code_type: Union[str, CodeType], date: str):
    # ensure code_type is CodeType
    if isinstance(code_type, str):
        code_type = CodeType[code_type]
    code_list = []
    if code_type == CodeType["FINANCIAL_FUTURE"]:
        code_list = ["IH_M0", "IF_M0", "IC_M0", "IM_M0"]
    elif code_type == CodeType["ETF"]:
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
    else:
        pass
    codes = [Code(x, code_type=code_type) for x in codes]
    return codes


def get_date_by_taskid(date_list, func_list, task_id):
    """
    根据date_list和task_id，返回当前任务需要处理的month
    """
    month_index = int(task_id)
    if (month_index < len(date_list)):
        pass
    else:
        print('not enough date_list', date_list, month_index)
        print('exit')
        sys.exit()
    return date_list[month_index], func_list


def get_date_func_by_taskid(date_list, func_list, task_id):
    """
    根据date_list, func_list和task_id，返回当前任务需要处理的month
    """
    month_index = int(round(int(task_id) // len(func_list)))
    func_index = int(round(int(task_id) % len(func_list)))
    if (month_index < len(date_list)):
        pass
    else:
        print('not enough date_list', date_list, month_index, func_index)
        print('exit')
        sys.exit()
    return date_list[month_index], func_list[func_index:func_index + 1]
