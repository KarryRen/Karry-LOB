# -*- coding: utf-8 -*-
# @author : MaMing
# @time   : 2024/03/01 10:25

""" The IO functions. """

from typing import Tuple, Union
import os
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from Ops import Exchange, Code, CodeType, Date

if os.access("/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering", os.W_OK):
    TensorEngineeringRootDir = f"/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering"
else:
    raise Exception(f"No write access. Please check.")


def get_file_path(code: Code, date: str, file_name: str) -> str:
    """ Get the full file path.

    :param code: the code
    :param date: the date
    :param file_name: the name of the file

    return:
        - the full path of the file

    """

    file_path = f"{TensorEngineeringRootDir}/{code}/{date}/{file_name}.h5"
    return file_path


def save_xarray_as_h5(code: Code, date: str, name: str, xray: xr.DataArray):
    """ Save the xarray to `.h5` file. 

    :param code: the code
    :param date: the date
    :param xray: the xray data

    """

    file_path = get_file_path(code, date, name)  # get the target save path
    xray.to_netcdf(file_path, mode="w", engine="h5netcdf", encoding={"__xarray_dataarray_variable__": {"zlib": True, "complevel": 5}})


def load_xarray_from_h5(code, date, name, path=None):
    if path:
        print('load path:', path)
    else:
        path = get_file_path(code, date, name)
    xray = xr.load_dataarray(path)
    return xray


def get_md_as_df_by_csv(code: Code, date: str, root_dir: str) -> Tuple[bool, pd.DataFrame]:
    """ Read the `.csv` format market data as dataframe.

    :param code: the code of market data
    :param date: the date of market data, format should be `yyyymmdd`
    :param root_dir: the root directory of market data

    return:
        - bool_md: read the market data right or not (md is existed or not)
        - df: the dataframe of market data

    """

    # ---- Construct the full path of `.csv` file ---- #
    full_raw_csv_file = f"{root_dir}/{code}/{date}.csv"

    # ---- Read the `.csv` file ---- #
    if os.path.exists(full_raw_csv_file):
        bool_md = True
        df = pd.read_csv(full_raw_csv_file)
        # test the tick Status
        assert (set(df["Status"].astype(int)) <= {int(0), int(1)}), "Status ERROR !!!"
        df.loc[:, "Status"] *= 3.0  # Set status from 1 to 3, 0 still be 0
    else:
        bool_md = False
        df = None
    return bool_md, df


def get_md_as_df_by_h5monthio(code, date, path_templete):
    file_path = path_templete % (str(code).replace('.', '')[:8], str(date)[:6])
    array_slice = np.hstack([np.arange(0, 144001, 10)[1:], np.arange(144001, 288002, 10)[1:]])
    bool_md = os.path.exists(file_path)
    df = None
    if bool_md:
        levels = 10
        features = ['Price', 'Volume']
        directions = ['Bid', 'Ask']
        lob_features = ["%s%s%d" % (direction, col, level) for direction in directions for level in range(1, levels + 1, 1)
                        for col in features] + ['Volume', 'Amount', 'LastPrice', "Status"]
        features = ['Price', 'Vol']
        lob_features_in_h5 = ["%s%s%d" % (direction, col, level) for direction in directions for level in range(1, levels + 1, 1)
                              for col in features] + ['CumVolume', 'CumAmount', 'LastPrice', "Status"]
        array_dict = {}
        with h5py.File(file_path, "r") as fp:
            for k, feature_in_h5 in enumerate(lob_features_in_h5):
                array_dict[lob_features[k]] = fp[feature_in_h5][date][:]
        array_values = np.stack(array_dict.values())
        array_values = array_values[:, array_slice]
        df = pd.DataFrame(array_values.T, columns=array_dict.keys())
    return bool_md, df


def get_md_as_df(code: Code, date: Union[Date, str]) -> Tuple[bool, pd.DataFrame]:
    """ Read the market data as dataframe.

    :param code: the code of market data
    :param date: the date of market data, format should be `yyyymmdd`

    return:
        - bool_md: read the market data right or not (md is existed or not)
        - df: the dataframe of market data

    """

    if code.exchange == Exchange["CFFEX"]:
        bool_md, df = get_md_as_df_by_csv(code, date, "/mnt/weka/home/test/maming_share/DataSet/DeepLobMd.2.1")
    elif code.code_type in [CodeType["ETF"], CodeType["ETF_OPTION"]]:
        bool_md, df = get_md_as_df_by_h5monthio(code, date, "/home/fisher_research/Data/50MS_DevMd/2019Data_%s/%s.h5")
        # 打个补丁. 2.x 适配 algo 之后, 可以去除这个补丁
        if code.code_type == CodeType["ETF"]:
            df.loc[:, "Status"] = 3.0
    elif code.exchange == Exchange["UNKNOWN"]:
        bool_md, df = get_md_as_df_by_csv(code, date, "/mnt/weka/home/test/maming_share/DataSet/DeepLobMd.2.1")
    else:
        raise ValueError(code)
    return bool_md, df
