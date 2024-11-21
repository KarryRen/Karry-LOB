# -*- coding: utf-8 -*-
# @Author  : Karry Ren
# @Time    : 2024/11/19 18:06

""" The util functions for tensor engineering. """

from typing import Union, List
import akshare as ak

from Ops import Code, CodeType


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
