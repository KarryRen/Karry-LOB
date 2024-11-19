# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/02/28 12:25
#
# pylint: disable=no-member

""" The base Class of TensorConstructionAlgo.

Classes in this file are really important. In practice, we will import a specific class,
  such as `from tensor_construction_base import ConstructionAlgoBase`, but in the same time
  all codes of Global Configs such as IO and Logging will be run !

"""

import os
import logging
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr

from Ops import Code, CodeType, Status, Date
from ..io import save_xarray_as_h5, get_md_as_df

# ---- Global IO Configs ---- #
global_io_config_dict = {}
if os.environ.get("realtime", "0") != "1":  # the global io config for `non-realtime` situation
    global_io_config_dict["1m_multilevel"] = {
        "base_dir": "/mnt/weka/home/test/maming_share/MultiLevel_Md_2310",
        "data_dir_prefix": "1M",
        "daily_stamp_num": 242,
        "io_type": "H5Month",
        "timestamps": {"intervals": "[['09:30:00', '11:30:00'], ['13:00:00', '15:00:00']]", "freq": "1min"}
    }
    global_io_config_dict["1m_volumedetail"] = {
        "base_dir": "/mnt/weka/home/test/maming_share/VolumeDetail_Md_2310",
        "data_dir_prefix": "1M",
        "daily_stamp_num": 242,
        "io_type": "H5Month",
        "timestamps": {"intervals": "[['09:30:00', '11:30:00'], ['13:00:00', '15:00:00']]", "freq": "1min"}
    }
else:  # the global io config for `realtime` situation
    global_io_config_dict["1m_multilevel"] = {
        "base_dir": "/home/fisher_research/Data/MultiLevel_Md_RealTime",
        "data_dir_prefix": "1M",
        "daily_stamp_num": 242,
        "io_type": "H5Month",
        "timestamps": {"intervals": "[['09:30:00', '11:30:00'], ['13:00:00', '15:00:00']]", "freq": "1min"}
    }
    global_io_config_dict["1m_volumedetail"] = {
        "base_dir": "/home/fisher_research/Data/VolumeDetail_Md_RealTime",
        "data_dir_prefix": "1M",
        "daily_stamp_num": 242,
        "io_type": "H5Month",
        "timestamps": {"intervals": "[['09:30:00', '11:30:00'], ['13:00:00', '15:00:00']]", "freq": "1min"}
    }

# ---- Global Logging Configs ---- #
logger_id = f"__call__{os.environ.get('realtime', '0')}__"  # define the logger id
call_logger = logging.getLogger(logger_id)  # set the call logger
file_handler = logging.FileHandler(f"{logger_id}.log")  # set the file handler
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))  # set the format of handler
call_logger.addHandler(file_handler)  # add the handler to the call logger


def valid_lob_df(lob_df: pd.DataFrame) -> bool:
    """ Valid the LOB DataFrame.

    Status==Status["TRADING"] means tick is OK.
    When tick is ok, 'BidVolume1' and 'AskVolume1' can't ALL be 0 (When the stop is made, AskVolume1 is 0).

    """

    if np.any(np.all(lob_df.loc[(lob_df["Status"] - Status["TRADING"].value).abs() < 1e-5, ["BidVolume1", "AskVolume1"]].values[:] < 1e-5, axis=1)):
        return False
    else:
        return True


class ConstructionAlgoBase:
    """ The base Class of Features.
    The data sources for the features are very diverse, and need to define: out_features, out_timestamps and out_ftype.

    """

    def __init__(self, out_coords: Dict[str, list], data_source: dict, version: Tuple[int, int, int, int, int],
                 init_data: float = 0.0, market_data_config: Optional[dict] = None, **kwargs):
        """ Init function of ConstructionAlgoBase.

        :param out_coords: the out coords of the constructed feature
            Such as {T:[time_period], L:range(1, 6), D:["Bid"], F:["Price"]} meaning (T, L, D, F) coord.
            The out_coords determines the shape of feature (self.xray).
        :param data_source: the source data dict of feature construction, format should be:
            {
                "source_1": {"param1": ..., "param2": ...},
                "source_2": {"param1": ..., "param2": ...},
                ...,
                "source_n": {"param1": ..., "param2": ...},
            } Such as {"LOB": {"features": ["Price", "Volume"], "levels": 5}} meaning source_data is
              `LOB` with 2 features and 5 levels.
        :param version: the version of this feature algorithm
            Because the feature algorithm is developing, so we need to note the version of it using
            the updating date 5 item tuple (Year, Month, Day, Hour, Minute) such as (2024, 2, 22, 0, 0).
        :param init_data: the initial data of feature array
            This param is really important ! It is the default num of the `nan` data.
        :param market_data_config: the config of market data

        """

        # ---- Set the params to self ---- #
        self.out_coords = out_coords
        self.data_source = data_source
        self.__version__ = version
        self.init_data = init_data

        # ---- Define new params ---- #
        # use the keys of coords to define the out feature type
        self.out_ftype = tuple(out_coords.keys())
        # define the xray and init it to be all `self.init_data`
        self.xray = None  # self.xray is the feature data container
        self.init_xray()

        # ---- Set the market data config ---- #
        if market_data_config is None:
            self.market_data_config = {}
        else:
            self.market_data_config = market_data_config
        if self.market_data_config.get("io") is None:  # `io` config default is `global_io_config_dict`
            self.market_data_config["io"] = global_io_config_dict
        if self.market_data_config.get("realtime") is None:  # `realtime` config is from os
            self.market_data_config["realtime"] = os.environ.get("realtime", "0") == "1"

    def cal_fea(self, code: Union[Code, List[Code]], date: Union[str, List[str]]) -> xr.DataArray:
        """ Feature calculation algorithm.

        :param code: the code of feature
            Could be 1 Code such as `Code("IH_M0")` or list of Codes such as `[Code("IH_M0"), Code("IF_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`

        """

        raise NotImplementedError("Please implement cal_fea function in the subclass !!!")

    def init_xray(self):
        """ Init the xray to be all `self.init_data`. The shape of `xray` is determined by out_coords. """

        self.xray = xr.DataArray(dims=self.out_ftype, coords=self.out_coords, data=self.init_data)

    def __call__(self, code: Union[Code, List[Code]], date: Union[str, List[str]], save: bool = True):
        """ The core feature calculation function, which includes the following steps:
            1. Test data.
            2. Use cal_fea interface to calculate features.

        :param code: the code of feature
            Could be 1 Code such as `Code("IH_M0")` or list of Codes such as `[Code("IH_M0"), Code("IF_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`

        return:
            - fea: the calculated feature

        """

        # ---- Step 1. Test the output feature type ---- #
        if list(self.out_ftype)[:1] == ["T"]:
            assert isinstance(code, Code), (code, self.out_ftype)
            assert isinstance(date, str), (date, self.out_ftype)
        else:  # out_ftupe must start with `T`
            raise ValueError(self.out_ftype)

        # ---- Step 2. Compute the feature ---- #
        try:  # try calculating
            fea = self.cal_fea(code, date)
        except Exception as e:  # have error, then log the error and raise error
            call_logger.error(f"code: {code}, date: {date}, error_msg: {e}, features: {self.out_coords['F']}")
            raise e
        else:  # have no error, save the feature
            if save:
                save_xarray_as_h5(code, date, type(self).__name__, self.xray)
        return fea

    def get_all_market_data(self, code: Union[Code, List[Code]], date: Union[str, List[str]]) -> Tuple[np.ndarray, Dict[str, xr.DataArray]]:
        """ Unified interface for obtaining all market data based on `self.data_source`.

        :param code: the code of feature
            Could be 1 Code such as `Code("IH_M0")` or list of Codes such as `[Code("IH_M0"), Code("IF_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`

        return:
            - ds_array_well: the flag array meaning ds_type is loaded right or not, shape=(ds_types_num)
            - ds_fea_dict: the dict of datasource feature with the format:
                {
                    "source_1": xray of source_1,
                    "source_2": xray of source_2,
                    ...,
                    "source_n": xray of source_n
                }

        """

        # ---- Get the data source type_num and define the empty data source array ---- #
        ds_types_num = len(self.data_source.keys())  # get the num of data source types
        ds_array_well = np.zeros(ds_types_num, dtype=np.bool_)  # the flag array meaning ds_type is loaded right or not, shape=(ds_types_num)
        ds_fea_dict = {}  # the feature dict of data source

        # ---- Get the valid status ---- #
        valid_status = False
        if isinstance(code, Code) and isinstance(date, (str, Date)):
            valid_status = True

        # ---- Load the market data ---- #
        if valid_status:
            for d, data_type in enumerate(self.data_source.keys()):
                ds_array_well[d], ds_fea_dict[data_type] = self.get_market_data(code, date, data_type, _private=True, **self.data_source[data_type])

        # ---- Mask the future information ---- #
        if "mask" in self.market_data_config.keys():
            mask_feas(ds_fea_dict, self.market_data_config["mask"])
        return ds_array_well, ds_fea_dict

    def get_static_data(self, code: Code, date: str) -> Dict[str, float]:
        """ Get the static data.

        :param code: the code of static data
        :param date: the date of static data

        return:
            - static_data: the static data dict

        """

        # ---- Define the static_data ---- #
        static_data = {
            "PriceTick": None,
            "VolumeMultiplier": None,
            "LowerLimitPrice": None,
            "UpperLimitPrice": None,
        }

        # ---- Set the static_data based on the code ---- #
        if code.code_type == CodeType["FINANCIAL_FUTURE"]:
            volume_multiplier_dict = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}
            static_data["PriceTick"] = 0.2
            static_data["VolumeMultiplier"] = volume_multiplier_dict[code.symbol[:2]]
        elif code.code_type == CodeType["ETF"]:
            static_data["PriceTick"] = 0.001
            static_data["VolumeMultiplier"] = 1.0
        else:
            raise NotImplementedError(code)
        return static_data

    def get_market_data(self, code: Union[Code, List[Code]], date: Union[str, List[str]], data_type: str, **kwargs) -> Tuple[bool, xr.DataArray]:
        """ 将 market data 的入口统一管理. 支持通过 mask 未来数据的方式, 确认算法没有用到未来数据。具体设置从 self.market_data_config 中获取

        :param code: the code of feature
            Could be 1 Code such as `Code("IH_M0")` or list of Codes such as `[Code("IH_M0"), Code("IF_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`
            ATTENTION: 目前日期需要是连续的交易日，如果不是的话，可能会出现未定义的错误
        :param data_type: the data_type of needed data, you have the following 2 choices now:
            - `LOB`: the Limit Order Book
            - `Trade`: the Trade Volume

        return:
            - bool_get_well: the flag meaning market data is loaded right or not
            - xray: the xarray of markey data

        """

        # ---- Test use the get_all_market_data() or not --- #
        assert kwargs.get("_private", False), "Please use get_all_market_data() to get market data rather than using get_market_data() directly !!!"

        # ---- Load the data based on the data_type and  ---- #
        if data_type == "LOB":  # for `LOB` data
            # test the code and date type
            assert isinstance(code, Code), f"Code: {code} is ERROR !"
            assert isinstance(date, (str, Date)), f"Date: {date} is ERROR !"
            # define the flag meaning get well or not and the empty xray
            bool_get_well, xray = False, None
            # read raw data
            bool_md, raw_data_df = get_md_as_df(code, date)
            if bool_md:  # md is existed
                if valid_lob_df(raw_data_df):  # pass the validation
                    bool_get_well = True
                    levels = kwargs.get("levels", 5)  # get the read levels, default is 5
                    features = kwargs.get("features", ["Price", "Volume"])  # get the read features, default is ["Price", "Volume"]
                    directions = ["Bid", "Ask"]  # set the direction
                    # get the lob_features
                    lob_features = [f"{direction}{col}{level:d}" for direction in directions for level in range(1, levels + 1, 1) for col in features]
                    # define the empty xray
                    xray = xr.DataArray(
                        dims=["timestamp", "direction", "level", "feature"],
                        coords={
                            "timestamp": list(range(28800)),
                            "direction": directions,
                            "level": np.arange(1, levels + 1, 1),
                            "feature": features
                        }
                    )
                    # set the data to the xray
                    xray.data[:] = raw_data_df[lob_features].values.reshape(xray.shape)
                else:  # not pass the validation
                    call_logger.error(f"LOB Status ERROR !!! code: {code}, date: {date}, data_type: {data_type}")
            else:  # md is not existed
                call_logger.error(f"LOB MD data is NOT EXISTED !!! code:{code}, date: {date}, data_type: {data_type}")
        elif data_type == "Trade":  # for `Trade` data
            # test the code and date type
            assert isinstance(code, Code), f"Code: {code} is ERROR !"
            assert isinstance(date, (str, Date)), f"Date: {date} is ERROR !"
            bool_get_well, xray = False, None
            # define the flag meaning get well or not and the empty xray
            bool_md, raw_data_df = get_md_as_df(code, date)
            if bool_md:  # md is existed
                if valid_lob_df(raw_data_df):  # pass the validation
                    bool_get_well = True
                    # get the trade_features
                    trade_features = kwargs.get("features", ["Status", "Volume", "Amount", "LastPrice"])
                    # define the empty xray
                    xray = xr.DataArray(
                        dims=["timestamp", "feature"],
                        coords={
                            "timestamp": list(range(28800)),
                            "feature": trade_features,
                        }
                    )
                    # set the data to the xray
                    xray.data[:] = raw_data_df[trade_features].values.reshape(xray.shape)
                else:  # not pass the validation
                    call_logger.error(f"TRADE Status ERROR !!! code: {code}, date: {date}, data_type: {data_type}")
            else:  # md is not existed
                call_logger.error(f"TRADE MD data is NOT EXISTED !!! code:{code}, date: {date}, data_type: {data_type}")
        else:
            raise ValueError(data_type)
        return bool_get_well, xray

    def check_data(self, code: str, date: str, data_dict: Dict):
        """ Check the data. """
        bool_check_well = True
        if {"LOB"} <= set(data_dict.keys()):
            bug_msg_l = check_multilevel_data(code, date, data_dict['LOB'], ['BidPrice', 'AskPrice', 'BidVolume', 'AskVolume'])
            bool_check_well = len(bug_msg_l) == 0
        return bool_check_well


def check_multilevel_data(code, date, fea, columns):
    """
    通过自写逻辑判断读取到的多档行情是否合理。
    输入：
        fea：已经读取后的多档行情数据，包括买卖的价量
        levels：检验前多少档位数据
    输出：
        质检有问题的信息列表，长度为0代表没有问题
    """
    bug_msg_l = []
    if {'BidPrice', 'AskPrice', 'BidVolume', 'AskVolume'} <= set(columns):
        # bid_volume_all = fea[fea.ix(features=["BidVol%d"%(level) for level in range(1, levels + 1)])].data[0,0,:,:] #(242,levels)
        # bid_price_all = fea[fea.ix(features=["BidPrice%d"%(level) for level in range(1, levels + 1)])].data[0,0,:,:]
        # ask_volume_all = fea[fea.ix(features=["AskVol%d"%(level) for level in range(1, levels + 1)])].data[0,0,:,:] #(242,levels)
        # ask_price_all = fea[fea.ix(features=["AskPrice%d"%(level) for level in range(1, levels + 1)])].data[0,0,:,:]
        bid_volume_all = fea.loc[:, 'Bid', :, "Volume"].data
        bid_price_all = fea.loc[:, 'Bid', :, "Price"].data
        ask_volume_all = fea.loc[:, 'Ask', :, "Volume"].data
        ask_price_all = fea.loc[:, 'Ask', :, "Price"].data

        for t in range(bid_volume_all.shape[0]):
            bid_volume = bid_volume_all[t, :]
            bid_price = bid_price_all[t, :]
            ask_volume = ask_volume_all[t, :]
            ask_price = ask_price_all[t, :]
            base_msg = f'code {code} date {date} timestamp index {t} bug massage : '
            # bid price降序且不出现中间有0
            if (np.diff(bid_price[bid_price > 1e-3]) >= 0).any():
                bug_msg_l.append(base_msg + 'bid price increase by level.')

            if (bid_price < 1e-3).any() and np.sum(bid_price > 1e-3) != (np.where(bid_price < 1e-3)[0][0]):  # 如果第一个<1e-3的值不是出现在最后
                bug_msg_l.append(base_msg + 'bid price has 0 interval.')

            # ask price生序且不出现中间有0
            if (np.diff(ask_price[ask_price > 1e-3]) <= 0).any():
                bug_msg_l.append(base_msg + 'ask price decrease by level.')

            if (ask_price < 1e-3).any() and np.sum(ask_price > 1e-3) != (np.where(ask_price < 1e-3)[0][0]):  # 如果第一个<1e-3的值不是出现在最后
                bug_msg_l.append(base_msg + 'ask price has 0 interval.')

            # bid price严格小于ask price (排除涨跌停的时候)
            if bid_price[0] >= ask_price[0] and ask_price[0] > 1e-3:  # 第二个用来判断涨停时ask为0，这时第一个一般都为True
                bug_msg_l.append(base_msg + f'bidprice0 {bid_price[0]} > askprice0 {ask_price[0]}.')

            # bid volume和ask volume不出现中间有0
            if (bid_volume < 1e-3).any() and np.sum(bid_volume > 1e-3) != (np.where(bid_volume < 1e-3)[0][0]):  # 如果第一个<1e-3的值不是出现在最后
                bug_msg_l.append(base_msg + 'bid volume has 0 interval.')

            if (ask_volume < 1e-3).any() and np.sum(ask_volume > 1e-3) != (np.where(ask_volume < 1e-3)[0][0]):  # 如果第一个<1e-3的值不是出现在最后
                bug_msg_l.append(base_msg + 'ask volume has 0 interval.')

            # bid volume和bid price有相同的档位
            if (np.logical_xor(bid_volume > 1e-3, bid_price > 1e-3)).any():
                bug_msg_l.append(base_msg + 'bid volume and price has different level.')

            # ask volume和ask price有相同的档位
            if (np.logical_xor(ask_volume > 1e-3, ask_price > 1e-3)).any():
                bug_msg_l.append(base_msg + 'ask volume and price has different level.')
    return bug_msg_l


def str_2_count(x):
    return int(int(x[:2]) * 3600 + int(x[3:5]) * 60 + int(x[6:8])) * 1e3 + int(x[-3:])


def mask_feas(fea_dict: Dict, market_data_config: Dict):
    """
    屏蔽未来信息.
    大于 date + timestamps的信息称为未来信息。 即，小于等于date + timestamps的信息可以正常使用；其余数据转化为nan.
    if else的逻辑分支，和self.get_market_data一致。

    self.get_market_data的返回值，某些是dataframe。 这种数据需要单独处理
    self.get_market_data的返回值，某些是Feature类。 这种数据可以统一处理

    """
    mask_date = market_data_config['date']
    mask_timestamps = market_data_config['timestamps']
    keys = fea_dict.keys()
    for key in keys:
        fea = fea_dict[key]
        print(key, type(fea))
        if isinstance(fea, pd.DataFrame):
            if key in ("Level10", "AlignedLevel10", "MinuteBar"):
                masked_fea = fea.copy()
                cond_1 = (masked_fea.TradingDate == mask_date) & ((masked_fea.TradingTime).apply(str_2_count) > str_2_count(mask_timestamps))
                cond_2 = masked_fea.TradingDate > mask_date
                # 现在把需要mask的数据都换成了nan。 会导致代码有问题
                # 原因是做了类型转换
                col_to_keep = ['Symbol', 'TradingDate', 'TradingTime', "UpdateTime"]
                for col in masked_fea.columns:
                    if col in col_to_keep:
                        # 不进行mask
                        pass
                    else:
                        # 根据type采用不同的mask方案
                        dtype = type(masked_fea[col].values[0])
                        if dtype == str:
                            masked_fea.loc[cond_1 | cond_2, col] = masked_fea[col].values[0]
                        elif dtype == np.float64:
                            masked_fea.loc[cond_1 | cond_2, col] = np.nan
                        elif dtype == np.int64:
                            masked_fea.loc[cond_1 | cond_2, col] = 0
                        elif dtype == np.datetime64:
                            pass
                        else:
                            print("dtype", dtype, dtype == object)
                            raise NotImplementedError()
                print("masked_fea:", masked_fea)
                fea_dict[key] = masked_fea
            elif key in ("Order", "Trade", "Cancel"):
                masked_fea = fea.copy()
                cond_1 = (masked_fea.TradingDate == mask_date) & (
                        (masked_fea.TradingTime).apply(str_2_count) > str_2_count(mask_timestamps))
                cond_2 = masked_fea.TradingDate > mask_date
                masked_fea = masked_fea.loc[~(cond_1 | cond_2), :]
                print("masked_fea:", masked_fea)
                fea_dict[key] = masked_fea
            elif key == "DailyStock":
                # DailyStock skip MAXUP MAXDOWN
                # 只有SYMBOL TRADING_DATE
                # 认为15:00才能得到数据
                masked_fea = fea.copy()
                if str_2_count(mask_timestamps) < str_2_count('15:00:00.000'):
                    cond = masked_fea.TRADING_DATE >= mask_date
                else:
                    cond = masked_fea.TRADING_DATE > mask_date
                col_to_keep = ['SYMBOL', 'TRADING_DATE', 'MAXUP', 'MAXDOWN']  # "TradingDate"
                for col in masked_fea.columns:
                    if col not in col_to_keep:
                        dtype = type(masked_fea[col].values[0])
                        if dtype == str:
                            masked_fea.loc[cond, col] = masked_fea[col].values[0]
                        elif dtype == np.float64:
                            masked_fea.loc[cond, col] = np.nan
                        elif dtype == np.int64:
                            masked_fea.loc[cond, col] = 0
                        elif dtype == np.datetime64:
                            pass
                        else:
                            print("dtype", col, dtype)
                            raise NotImplementedError()
                fea_dict[key] = masked_fea
            else:
                raise ValueError(key)
        else:
            raise NotImplementedError(type(fea))
