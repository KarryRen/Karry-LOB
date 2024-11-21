# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/03/15 11:25

""" The datatype Class.

Manage the datatype more regularly and safely.

"""

from enum import Enum
from datetime import datetime
from typing import Union


class Status(Enum):
    """ The CodeType Class. Inherit Enum and Enum the `str` CodeType to `int`.

    You can use the `Status[str_status]` or `Status[int_status]` to get the int_status.

    """

    UNKNOWN = 0
    CLOSED = 1  # 收盘
    FUSE = 2  # 集合竞价
    TRADING = 3  # 连续竞价
    PAUSED = 4  # 休市
    SUSPENDED = 5  # 停牌
    VOLATILITY_INTERRUPT = 6  # 波动性中断
    RECOVERABLE_MELTDOWN = 7  # 可恢复熔断
    IRRECOVERABLE_MELTDOWN = 8  # 不可恢复熔断


class CodeType(Enum):
    """ The CodeType Class. Inherit Enum and Enum the `str` CodeType to `int`.

    You can use the `CodeType[str_code_type]` or `CodeType[int_status]` to get the int_code_type.

    """

    UNKNOWN = 0
    STOCK = 1
    ETF = 2
    REPO = 3
    INDEX = 4
    BOND = 5
    STOCK_OPTION = 6
    ETF_OPTION = 7
    COMMODITY_FUTURE = 8
    FINANCIAL_FUTURE = 9
    COMMODITY_OPTION = 10
    FINANCIAL_OPTION = 11
    FUND = 12
    __LAST = 13

    def __str__(self):
        return self.name


class Exchange(Enum):
    """ The Exchange Class. Inherit Enum and Enum the `str` Exchange Type to `int`.

    You can use the `Exchange[str_exchange]` or `Exchange[int_status]` to get the int_exchange.

    """

    UNKNOWN = 0
    SHSE = 1
    SZSE = 2
    CFFEX = 3
    SHFE = 4
    DCE = 5
    CZCE = 6
    INE = 7
    SGE = 8
    HKEX = 9
    SGX = 10  # 新加坡交易所
    WXBXG = 11  # 无锡不锈钢电子交易中心
    BINANCE = 12  # 币安
    BSE = 13  # 北京证券交易所
    CBOT = 14  # 芝加哥期货交易所
    CME = 15  # 芝加哥商品交易所
    COMEX = 16  # 纽约商品交易所COMEX分部

    def __str__(self):
        return self.name


def exchange_transform(ex: str) -> str:
    """ The transform function of exchange.

    :param ex: the type of exchange, such as `SZ`

    returns:
        - ex_trans: the transformed type of exchange.

    """

    if ex in ["SZ", "sz"]:
        ex_trans = "SZSE"
    elif ex in ["SH", "sh"]:
        ex_trans = "SHSE"
    else:
        ex_trans = ex
    return ex_trans


class Code:
    """ Code Class. Manage the Code more regularly and safely. """

    def __init__(self, code: str = None, symbol: str = None, exchange: str = None,
                 code_type: Union[str, CodeType] = "UNKNOWN", **kwargs):
        """ Analyze the code, symbol and exchange.

        :param code: the code (Unique identification of Stocks), such as "600519.SH", "IH_M0"
        :param symbol: the symbol, such as "600519", "IH_M0"
        :param exchange: the exchange, the "SH", "SZ"
        :param code_type: the type of code

        """

        if code is None:  # code is None, use symbol and exchange to get code.
            if symbol is not None and exchange is not None:
                self.symbol = symbol
                self.exchange = Exchange[exchange]
            else:  # symbol and exchange can't be None.
                raise ValueError((code, symbol, exchange))
        else:  # code is not None
            if symbol is not None and exchange is not None:  # have the symbol and exchange
                self.symbol = symbol
                self.exchange = Exchange[exchange]
            else:  # don't have the symbol or exchange
                if "." in code:  # have exchange in code, such as stock
                    self.symbol = code.split(".")[0]
                    self.exchange = Exchange[exchange_transform(code.split(".")[1])]
                else:  # have no exchange in code, such as future
                    self.symbol = code
                    self.exchange = Exchange["UNKNOWN"]
        self.code = self.__str__()

        # 暂时不自动推断。但是其实也可以自动推断
        self.code_type = CodeType[code_type] if isinstance(code_type, str) else code_type

        # analyse the kwargs
        self.__info = kwargs
        assert (set(self.__info.keys()) <= {"optiontype", "underlying", "strike_price", "expire_date"})

    def get(self, key):
        return self.__info.get(key)

    def __str__(self):
        if self.exchange == Exchange["UNKNOWN"]:
            return self.symbol
        else:
            return f"{self.symbol}.{self.exchange}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        value = hash(self.code) + hash(self.symbol) + hash(self.exchange)
        for key, values in self.__info.items():
            value += hash(key)
            value += hash(values)
        return value

    def __eq__(self, others):
        return (self.code == others.code) and (self.symbol == others.symbol) and (
                self.exchange == others.exchange) and (self.__info == others.__info)

    def __gt__(self, other):
        return self.code > other.code

    def __lt__(self, other):
        return self.code < other.code


class Date(datetime):
    """ The Date class for manage the date type variables. """

    def __new__(cls, *args, **kwargs):
        """ Change the constructed function of `datetime` class. """
        if len(args) == 1:  # only support the 1 variable
            date = args[0]
            # date is datetime type
            if isinstance(date, datetime):
                return super().__new__(cls, date.year, date.month, date.day, )
            # date is bytes type
            elif isinstance(date, bytes) and len(date) == 10 and 1 <= ord(date[2:3]) & 0x7F <= 12:
                return super().__new__(cls, date)
            # date is str type (the most common situation we faced during research)
            elif isinstance(date, str):
                if "-" in date:  # date is the format of `yyyy-mm-dd` length is 10
                    args = tuple(map(int, date[:10].split("-")))  # transfer to (yyyy, mm, dd)
                else:  # date is the format of `yyyymmdd` length is 8
                    args = tuple(map(int, (date[:4], date[4:6], date[6:8])))  # transfer to (yyyy, mm, dd)
                return super().__new__(cls, *args, **kwargs)  # construct the datetime variable
        else:
            raise ValueError(args)

    def __str__(self) -> str:
        return self.strftime("%Y%m%d")

    def __repr__(self) -> str:
        return self.__str__()

    def format_month(self) -> str:
        return self.__str__()[:6]
