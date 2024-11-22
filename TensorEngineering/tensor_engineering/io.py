# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/03/01 10:25

""" The IO functions. """

from typing import Tuple, Union
import os
import pandas as pd
import xarray as xr

from Ops import Exchange, Code, Date

if os.access("/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering", os.W_OK):
    TensorEngineeringRootDir = "/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering"
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


def save_xarray_as_h5(code: Code, date: str, name: str, xray: xr.DataArray) -> None:
    """ Save the xarray to `.h5` file.

    :param code: the code
    :param date: the date
    :param name: the name of the feature
    :param xray: the xray data

    """

    file_path = get_file_path(code, date, name)  # get the target save path
    xray.to_netcdf(file_path, mode="w", engine="h5netcdf", encoding={"__xarray_dataarray_variable__": {"zlib": True, "complevel": 5}})


def load_xarray_from_h5(code: Code, date: str, name: str, path: str = None) -> xr.DataArray:
    """ Load the xarray from `.h5` file.

    :param code: the code
    :param date: the date
    :param name: the name of the feature
    :param path: the direct path of the file.

    """

    if path:
        print(f"Load the xarray from .h5 path: {path} directly !")
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
        assert (set(df["Status"].astype(int)) <= {int(0), int(1)}), "Status ERROR !!!"  # test the tick Status
        df.loc[:, "Status"] *= 3.0  # Set status from 1 to 3, 0 still be 0. 3 means `连续竞价`
    else:
        bool_md = False
        df = None
    return bool_md, df


def get_md_as_df(code: Code, date: Union[Date, str]) -> Tuple[bool, pd.DataFrame]:
    """ Read the market data as dataframe.

    :param code: the code of market data
    :param date: the date of market data, format should be `yyyymmdd`

    return:
        - bool_md: read the market data right or not (md is existed or not)
        - df: the dataframe of market data

    """

    if code.exchange == Exchange["UNKNOWN"]:
        bool_md, df = get_md_as_df_by_csv(code, date, "/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/LOB_Raw")
    else:
        raise ValueError(code)
    return bool_md, df
