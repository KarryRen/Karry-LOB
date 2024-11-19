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


def get_code_list(code_type: Union[str, CodeType]) -> List[Code]:
    """ Get the code list based on code_type. """

    # Ensure code_type is CodeType
    if isinstance(code_type, str):
        code_type = CodeType[code_type]

    # Get the detailed code list, and transfer to Code
    if code_type == CodeType["FINANCIAL_FUTURE"]:
        code_list = ["IF_M0"]
    else:
        raise TypeError(code_type)
    code_list = [Code(x, code_type=code_type) for x in code_list]
    return code_list
