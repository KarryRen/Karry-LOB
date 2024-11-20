# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/02/28 12:25

""" The base Class of TensorConstructionAlgo.

Classes in this file are really important. In practice, we will import a specific class,
    such as `from tensor_construction_base import ConstructionAlgoBase`, but in the same time
    all codes of Global Configs such as Logging will be run !

"""

import os
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
import numpy as np
import pandas as pd
import xarray as xr

from Ops import Code, CodeType, Status, Date
from ..io import save_xarray_as_h5, get_md_as_df

# ---- Global Logging Configs ---- #
logger_id = f"__call__{os.environ.get('realtime', '0')}__"  # define the logger id
call_logger = logging.getLogger(logger_id)  # set the call logger
call_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(f"{logger_id}.log")  # set the file handler
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))  # set the format of handler
call_logger.addHandler(file_handler)  # add the handler to the call logger


def valid_lob_df(lob_df: pd.DataFrame) -> bool:
    """ Valid the LOB DataFrame.

    Status == Status["TRADING"] means tick is OK (is TRADING).
    When tick is ok, 'BidVolume1' and 'AskVolume1' can't ALL be 0 (When the stop is made, AskVolume1 is 0).
    So, if any tick Status is TRADING but `BidVolume1` and `AskVolume1` are both 0, the data has errors!

    """

    if np.any(np.all(lob_df.loc[(lob_df["Status"] - Status["TRADING"].value).abs() < 1e-5, ["BidVolume1", "AskVolume1"]].values[:] < 1e-5, axis=1)):
        return False
    else:
        return True


class ConstructionAlgoBase:
    """ The base Class of Construction. """

    def __init__(
            self, out_coords: Dict[str, list], data_source: dict, version: Tuple[int, int, int, int, int],
            init_data: float = 0.0, market_data_config: Optional[dict] = None, **kwargs
    ):
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
        if self.market_data_config.get("realtime") is None:  # set the `realtime` config from os params
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
            Could be 1 Code such as `Code("IF_M0")` or list of Codes such as `[Code("IF_M0"), Code("IC_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`

        return:
            - fea: the calculated feature

        """

        # ---- Test the output feature type ---- #
        if list(self.out_ftype)[:1] == ["T"]:
            assert isinstance(code, Code), (code, self.out_ftype)
            assert isinstance(date, str), (date, self.out_ftype)
        else:  # out_ftype must start with `T`
            raise ValueError(self.out_ftype)

        # ---- Compute the feature ---- #
        try:  # try calculating
            fea = self.cal_fea(code, date)
            call_logger.info(f"INFO: code: {code}, date: {date}, feature: `{type(self).__name__}` calculate successfully !")
        except Exception as e:  # have error, then log the error and raise error
            call_logger.error(f"ERROR: code: {code}, date: {date}, error_msg: {e}, features: {self.out_coords['F']}")
            raise e
        else:  # have no error, save the feature
            if save:
                save_xarray_as_h5(code, date, type(self).__name__, self.xray)
                call_logger.info(f"INFO: code: {code}, date: {date}, feature: `{type(self).__name__}` save successfully !")
        return fea

    def get_all_market_data(self, code: Union[Code, List[Code]], date: Union[str, List[str]]):
        """ Unified interface for obtaining all market data based on `self.data_source`.

        :param code: the code of feature
            Could be 1 Code such as `Code("IF_M0")` or list of Codes such as `[Code("IF_M0"), Code("IC_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`

        :return:
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
        return ds_array_well, ds_fea_dict

    @staticmethod
    def get_market_data(code: Union[Code, List[Code]], date: Union[str, List[str]], data_type: str, **kwargs) -> Tuple[bool, xr.DataArray]:
        """ Unified market data interface for ONE data_type

        :param code: the code of feature
            Could be 1 Code such as `Code("IF_M0")` or list of Codes such as `[Code("IF_M0"), Code("IC_M0"), ...]`
        :param date: the date of feature
            Could be 1 date such as `yyyymmdd` or list of dates such as `[yyyymmdd1, yyyymmdd2, ...]`
        :param data_type: the data_type of needed data, you have the following 1 choice now:
            - `LOB`: the Limit Order Book

        :return:
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
            # read the market data as df
            bool_md, raw_md_df = get_md_as_df(code, date)
            if bool_md:  # md is existed
                if valid_lob_df(raw_md_df):  # pass the validation
                    bool_get_well = True
                    levels = kwargs.get("levels", 5)  # get the read levels, default is 5
                    features = kwargs.get("features", ["Price", "Volume"])  # get the read features, default is ["Price", "Volume"]
                    directions = ["Bid", "Ask"]  # set the direction
                    # get the lob_features list
                    lob_features = [f"{d}{f}{l:d}" for d in directions for l in range(1, levels + 1, 1) for f in features]
                    # define the empty xray
                    xray = xr.DataArray(
                        dims=["timestamp", "direction", "level", "feature"],
                        coords={
                            "timestamp": range(0, 28800),
                            "direction": directions,
                            "level": np.arange(1, levels + 1, 1),
                            "feature": features
                        }
                    )
                    # set the data to the xray
                    xray.data[:] = raw_md_df[lob_features].values.reshape(xray.shape)
                else:  # not pass the validation
                    call_logger.error(f"ERROR: LOB Status ERROR !!! code: {code}, date: {date}, data_type: {data_type}")
                    raise Exception("LOB Status ERROR !!!")
            else:  # md is not existed
                call_logger.error(f"ERROR: LOB MD data is NOT EXISTED !!! code:{code}, date: {date}, data_type: {data_type}")
                raise Exception("LOB MD data is NOT EXISTED !!!")
        else:
            raise ValueError(data_type)
        return bool_get_well, xray
