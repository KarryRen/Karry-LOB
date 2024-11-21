# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" The data dict for deeplob operation.

All features of one date of one future should from TensorEngineering.
We will not use the dataset to read `.h5` raw file directly, but use this deeplob datadict to do some
    preprocess. You can follow the `config` files to set the param of data dict.

The prepared data after Tensor Engineering should have the following root structure:
    data_root_path/
    ├── future_type1 (such as `IF_MO`)
        ├── date_1 (format is `yyyymmdd`, such as `20220204`)
            ├── label1.h5
            ├── feature1.h5
            ├── feature2.h5
            └── ...
        ├── date_2
        ├── ...
        └── date_n
    ├── future_type2 (such as `IC_MO`)
    ├── ...
    └── future_typen (such as `IM_MO`)

"""

import os
from typing import List, Dict, Union
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict

from Ops import Date


class IOBase:
    """ The IO Base Class. """

    def __init__(self, root_dir: str, **kwargs):
        """ Init of Data IO Base Class.

        :param root_dir: the root directory of data.

        """

        self.root_dir = root_dir

    @staticmethod
    def apply_transform(array: Union[np.ndarray, xr.DataArray], config: dict) -> xr.DataArray:
        """ Transform the array after reading.

        :param array: the read array
        :param config: the config file of data. Might be the following format:
            - Empty dict {} for no config features and labels.
            - {"file_name":xx, "transform_type":xxx} for features having config .

        """

        assert isinstance(array, xr.DataArray)
        return IOBase.apply_transform_from_xarray(array, config)

    @staticmethod
    def apply_transform_from_xarray(xarray: xr.DataArray, config: dict) -> xr.DataArray:
        """ Apply the transform operations of `config` to xarray.

        :param xarray: the array to do the transformation
        :param config: the config of transformation

        return:
            - the transformed array data

        """

        if config == {}:  # return the data of xarray directly if config is empty
            return xarray
        else:  # config is not empty, need transforming
            loop_xarray = xarray.copy()  # for loop to transform (the transform config might have more than 1 item)
            for config_key, config_params in config.items():
                if config_key == "file_name":  # `file_name` is not transformation
                    pass
                elif config_key == "boxcox":  # the `boxcox` transform
                    raise NotImplementedError()
                elif config_key == "slice":  # the `slice` transform
                    if isinstance(config_params[0], (int, np.int64, np.int32)):
                        loop_xarray = loop_xarray.isel(F=config_params)
                    elif isinstance(config_params[0], str):
                        loop_xarray = loop_xarray.sel(F=config_params)
                    else:
                        raise ValueError()
                elif config_key == "scale":  # the `scale` transform
                    loop_xarray.data[:] = loop_xarray.data[:] * config_params
                elif config_key == "shift":  # the `shift` transform
                    shift_num, padding = config_params
                    loop_xarray = loop_xarray.shift({"T": shift_num}, fill_value=padding)
                elif config_key == "exchange_pv":  # exchange p/v at bid/ask
                    raise NotImplementedError()
                elif config_key == "diff_p":
                    raise NotImplementedError()
                elif config_key == "cum_v":
                    loop_xarray.loc[{"F": "Volume"}] = loop_xarray.sel({"F": "Volume"}).cumsum("L")
                elif config_key == "transpose":  # the `transpose` transform
                    if isinstance(config_params[0], int):
                        if len(loop_xarray.dims) == len(config_params):  # one code situation
                            loop_xarray = loop_xarray.transpose(*[loop_xarray.dims[k] for k in config_params])
                        else:  # multi code situation
                            dims_exclude_code = [x for x in loop_xarray.dims if x != "code"]
                            if len(dims_exclude_code) == len(config_params):
                                new_dims = [dims_exclude_code[k] for k in config_params]
                                new_dims.insert(loop_xarray.dims.index("code"), "code")
                                loop_xarray = loop_xarray.transpose(*new_dims)
                            else:
                                raise NotImplementedError()
                    elif isinstance(config_params[0], str):
                        if len(loop_xarray.dims) == len(config_params):  # one code situation
                            loop_xarray = loop_xarray.transpose(*config_params)
                        else:  # multi code situation
                            dims_exclude_code = [x for x in loop_xarray.dims if x != "code"]
                            if len(dims_exclude_code) == len(config_params):
                                new_dims = ["code"] + list(config_params)
                                loop_xarray = loop_xarray.transpose(*new_dims)
                            else:
                                raise NotImplementedError()
                    else:
                        raise NotImplementedError()
                elif config_key == "exp_atten":  # the `exp_atten` transform
                    f_list = list(loop_xarray.coords["F"].values)
                    assert "FillNanSteps" in f_list, "`FillNanSteps` is not here, can't use exp_atten !"
                    for f in f_list:
                        if f.startswith("Volume"):
                            loop_xarray.loc[{"F": f}] = loop_xarray.loc[{"F": f}] * np.exp(config_params[0] * -loop_xarray.loc[{"F": "FillNanSteps"}])
                elif config_key == "padding_zeros":
                    num_of_padding_zeros = config_params[0]  # get the number of padding zeros
                    coords_dict = {}
                    dims_list = loop_xarray.dims  # get the right sequence of dims
                    for dim in dims_list:  # get data of dims
                        coords_dict[dim] = loop_xarray.coords[dim].data
                    coords_dict["F"] = [f"padding_zeros_{x}" for x in range(num_of_padding_zeros)]  # change the F
                    zeros_xarray = xr.DataArray(dims=tuple(coords_dict.keys()), coords=coords_dict, data=0.0)  # construct the zeros
                    loop_xarray = xr.concat([loop_xarray, zeros_xarray], dim="F")  # cat the zeros to loop xarray
                else:
                    raise ValueError(config_key)
            return loop_xarray

    def read_as_xarray(self, code: Union[str, List[str]], date: str, data_name: str, config: dict, dtype: str = "np.float64") -> xr.DataArray:
        """ Read data as xarray.

        :param code: the code of subject asset, for futures it might be `IF_M0` or `IC_M0`.
        :param date: the data, format must be `yyyymmdd`
        :param data_name: the name of data, name of label or features
        :param config: the config file of data. Might be the following format:
            - Empty dict {} for no config features and labels.
            - {"file_name":xx, "transform_type":xxx} for features having config.
        :param dtype: the type of data in target xarray, you have only two choices:
            - `np.float64`: default
            - `np.float32`

        """

        # ---- Step 1. Read the raw array and print the information ---- #
        raw_xarray = self._read_as_xarray(code, date, data_name)

        # ---- Step 2. Transform the array based on the config ---- #
        xarray = IOBase.apply_transform(raw_xarray, config)

        # ---- Step 3. Change the dtype of xarray ---- #
        if dtype == "np.float64":
            xarray = xarray.astype(np.float64)
        elif dtype == "np.float32":
            xarray = xarray.astype(np.float32)
        else:
            raise TypeError(dtype)
        return xarray

    def data_exists(self, code: Union[str, List[str]], date: str, data_name: str) -> bool:
        """ Judge whether data exist or not.

        :param code: the symbol of subject asset, for futures it might be `IF_M0` or `IC_M0`.
        :param date: the data, format must be `yyyymmdd`
        :param data_name: the name of data, name of label or features

        """
        raise NotImplementedError("The `data_exists()` function must implemented in subclasses !!")

    def _read_as_array(self, code: Union[str, List[str]], date: str, data_name: str) -> np.ndarray:
        """ Read data as array interface, implemented in subclasses.

        :param code: the symbol of subject asset, for futures it might be `IF_M0` or `IC_M0`.
        :param date: the data, format must be `yyyymmdd`
        :param data_name: the name of data, name of label or features

        """
        raise NotImplementedError("The `_read_as_array()` function must implemented in subclasses !!")

    def _read_as_xarray(self, code: Union[str, List[str]], date: str, data_name: str) -> xr.DataArray:
        """ Read data as xarray interface, implemented in subclasses.

        :param code: the symbol of subject asset, for futures it might be `IF_M0` or `IC_M0`.
        :param date: the data, format must be `yyyymmdd`
        :param data_name: the name of data, name of label or features

        """
        raise NotImplementedError("The `_read_as_xarray()` function must implemented in subclasses !!")


class IOH5py(IOBase):
    """ The IO Class for h5py `.h5` files. """

    def __init__(self, root_dir: str, **kwargs):
        """ Init the IOH5py Class. """
        super(IOH5py, self).__init__(root_dir=root_dir, **kwargs)

    def data_exists(self, code: Union[str, List[str]], date: str, data_name: str) -> bool:
        """ Judge whether data exists or not. """

        if isinstance(code, str):
            data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
            return os.path.exists(data_full_path)
        else:
            raise NotImplementedError()

    def _read_as_array(self, code: str, date: Date, data_name: str) -> np.ndarray:
        """ Read the data as array. """

        data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
        with h5py.File(data_full_path, "r") as feature_file:
            data_array = feature_file["data"][()]
        return data_array


class IOTensor(IOBase):
    """ The IO Class for TensorEngineering `.h5` files. """

    def __init__(self, root_dir: str, **kwargs):
        super(IOTensor, self).__init__(root_dir=root_dir, **kwargs)

    def data_exists(self, code: Union[str, List[str]], date: Date, data_name: str) -> bool:
        """ Judge whether data exists or not. """

        if isinstance(code, str):  # just one code, if exist then exist
            data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
            return os.path.exists(data_full_path)
        elif isinstance(code, list):  # the code list, must all exist then exist
            bool_exists = True
            codes = code.copy()
            for code in codes:
                data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
                if not os.path.exists(data_full_path):
                    bool_exists = False
                    break
            return bool_exists
        else:
            raise NotImplementedError()

    def _read_as_array(self, code: Union[str, List[str]], date: str, data_name: str) -> np.ndarray:
        """ Read the `.h5` file to np.ndarray.

        return:
            - if code is str: return (timestamps, *tensor.shape)
            - if code is List[str]: return (code_num, timestamps, *tensor.shape)

        """

        if isinstance(code, str):
            data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
            with h5py.File(data_full_path, "r") as feature_file:
                data_array = feature_file["__xarray_dataarray_variable__"][()]
            return data_array
        else:
            data_array = []
            for one_code in code:
                data_full_path = f"{self.root_dir}/{one_code}/{date}/{data_name}.h5"
                with h5py.File(data_full_path, "r") as feature_file:
                    one_data_array = feature_file["__xarray_dataarray_variable__"][()]
                data_array.append(one_data_array)
            data_array = np.stack(data_array)
            return data_array

    def _read_as_xarray(self, code: Union[str, List[str]], date: str, data_name: str) -> xr.DataArray:
        """ Read the `.h5` file to xr.DataArray.

        return:
            - if code is str: return (timestamps, *tensor.shape)
            - if code is List[str]: return (code_num, timestamps, *tensor.shape)

        """

        if isinstance(code, str):
            data_full_path = f"{self.root_dir}/{code}/{date}/{data_name}.h5"
            data_xarray = xr.load_dataarray(data_full_path)
            return data_xarray
        else:
            data_xarray = []
            for one_code in code:
                data_full_path = f"{self.root_dir}/{one_code}/{date}/{data_name}.h5"
                one_data_xarray = xr.load_dataarray(data_full_path)
                data_xarray.append(one_data_xarray)
            data_xarray = xr.concat(data_xarray, pd.Index(code, name="code"))
            return data_xarray


class DataDict(defaultdict):
    """ The DataDict Class (A great dict to use).

    {
        date_1: {
            data_name_1: array_1, data_name_2: array_2, ..., data_name_n: array_n
        },
        date_2: {...}, ...,
    }

    """

    def __init__(self):
        super().__init__(dict)

    def __iter__(self):
        """ overwrite __iter__ for matching ray_ds """
        for key, data in super().items():
            metadata = {b"date": key.encode()}
            yield data, metadata

    def show(self):
        """ overwrite show to print """
        for date in self.keys():
            for name in self.__getitem__(date).keys():
                print(date, name, self.__getitem__(date).__getitem__(name).shape)


def get_io(io_type: str, **kwargs) -> IOBase:
    """ The interface of io factory.

    :param io_type: the io type, you have only to choices now:
        - `h5py` : Get data from `deeplob_feature_engineering`.
        - `TensorEngineering` : Get data from `TensorEngineering`.

    return:
        - the instance of IOBase

    """

    if io_type == "h5py":
        return IOH5py(**kwargs)
    elif io_type == "TensorEngineering":
        return IOTensor(**kwargs)
    else:
        raise ValueError(io_type)


def gen_data_dict_config(needed_data: Dict) -> Dict[str, dict]:
    """ Generate the config for data dict of DeepLOB dataset.

    :param needed_data: the dict of needed data (The MOST important param of Dataset !)
        The format of this dict could be TWO types or MIXED:
            - Dict of [str, list], the item in list should be the name of data
                {
                    "data_type_1": [data_1_1, data_1_2, ...],
                    "data_type_2": [data_2_1, data_2_2, ...],
                    ...,
                    "data_type_n": [data_n_1, data_n_2, ...]
                } The keys of this dict are explicit declaration of data.
            - Dict of dict, the item dict is the detail config of data
                {
                    "data_type_1": {"data_1_1": {config_of_f11}, "data_1_2": {config_of_f12}, ...},
                    "data_type_2": {"data_2_1": {config_of_f21}, "data_2_2": {config_of_f22}, ...},
                    ...,
                    "data_type_n": {"data_n_1": {config_of_fn1}, "data_n_2": {config_of_fn2}, ...},
                } The keys of this dict are explicit declaration of data.

    return:
        - data_dict_config: the config of data_dict (extracted from data), the format should be
            {
                "data_1": {config_of_data_1},
                "data_2": {config_of_data_2},
                ...,
                "data_n": {config_of_data_n}
            } The keys of this dict are names of data.

    """

    # ---- Define the empty data dict config ---- #
    data_dict_config = {}

    # ---- Make the data config ---- #
    assert isinstance(needed_data, dict), "Now the DeepLOB datadict only support `dict` type needed_data !!"
    for data_type, data_values in needed_data.items():
        if isinstance(data_values, list):  # the first type of data_values -> list
            for data_name in data_values:
                data_dict_config[data_name] = {}  # define as the empty dict
        elif isinstance(data_values, dict):  # the second type of data_values -> dict
            for key in data_values.keys():  # check config same
                if key in data_dict_config.keys():
                    assert data_dict_config[key] == data_values[key], "Same data name but different config !!"
            data_dict_config.update(data_values)  # insert all dict to data_dict_config
        else:
            raise TypeError(data_type)
    return data_dict_config


def gen_data_dict(
        data_root_path_dict: Dict[str, str], codes: Union[str, List[str]],
        dates: List[str], data_dict_config: Dict[str, dict], verbose: bool = False
) -> DataDict:
    """ Generate the data dict, the cache of all needed data.

    :param data_root_path_dict: the dict of all possible data root path, the format should be
        {
            "data_root_path_1": "io_type_1",
            "data_root_path_2": "io_type_2",
            ...,
            "data_root_path_n": "io_type_n"
        } Detail directory structure refer to the top comment.
    :param codes: the list of code you want to model, such as IF_M0
    :param dates: the SORTED list of dates, such as [`yyyymmdd1`, `yyyymmdd2`, ...]
    :param data_dict_config: the config of data_dict generated by `gen_data_dict_config`, the format should be
        {
            "label_xx": {},
            "feature_1_1":
            {
                "file_name": xx,
                "`slice` or `boxcox`, ...": xx
            }, ...,
        } The keys of this dict are names of data. Have no config for feature, just is {}.
    :param verbose: a boolean flag indicating whether to enable verbose output, if set to True,
        additional information will be printed.

    return:
        - data_dict: the dict cache of all needed data, format is
            {
                "date_0": {label_xx: array_0, feature_xx: array_a, ... ,},
                "date_1": {label_xx: array_0, feature_xx: array_a, ... ,},
                ...,
                "date_n": {label_xx: array_0, feature_xx: array_a, ... ,}
            } Two layer dict, the key of the first layer is `date` while the key of the second layer is `data_name`.
    """

    # ---- Define the empty data dict ---- #
    data_dict = DataDict()

    # ---- Load all data ---- #
    for date in dates:  # for loop all dates
        for data_name, data_config in data_dict_config.items():  # for loop all data
            data_key = f"{date}/{data_name}"  # the key of data (unique identification)
            data_file_name = data_config.get("file_name", data_name)  # the file name of data, you can set in config or use default `data_name`
            # for loop all possible root path to find data
            found_data_flag = False  # the flag to indicate data found or not
            for root_path, io_type in data_root_path_dict.items():  # check the data from all data root path (one by one)
                data_io = get_io(io_type=io_type, root_dir=root_path)  # get the io interface
                if data_io.data_exists(codes, date, data_file_name):
                    found_data_flag = True  # find the data
                    xarray = data_io.read_as_xarray(codes, date, data_file_name, data_config)  # read it
                    data_dict[date][data_name] = xarray  # set it to the data dict
                    if verbose or (np.sum(np.isnan(xarray.data)) > 0):  # check data
                        print(f"Data Check: "
                              f"{codes}, {date}, {data_name:20}, "
                              f"nan_num={np.sum(np.isnan(xarray.data))}, "
                              f"nan_mean={np.nanmean(xarray.data)}, "
                              f"nan_std={np.nanstd(xarray.data)}")
                        # print successful info
                        print(f"Successfully read data dict: key={date, data_name}, shape={xarray.shape}, path={root_path}\n")
                    # break directly, which means just read from the data in first path
                    break
            # not found situation
            assert found_data_flag, f"{data_key} is not exist! Please Check your features! {data_root_path_dict} {data_dict_config}"
    return data_dict


def group_data_dict(data_dict: DataDict, group_config: Dict[str, List[str]], data_type: str):
    """ Group the data of one type.

    :param data_dict: the dict cache of all needed `label` and `features`, format is:
        {
            "date_0": {label_xx: array_0, feature_xx: array_a, ... ,},
            "date_1": {label_xx: array_0, feature_xx: array_a, ... ,},
            ...,
            "date_n": {label_xx: array_0, feature_xx: array_a, ... ,}
        } Two layer dict, key of the first layer is date while key of the second layer id data_name.
    :param group_config: the config of grouping, format is:
        {
            group_name_1(data_type_1, such as `(S, L, D, F)_1`): [data_name_1, data_name_2, ...,],
            group_name_2(data_type_2, such as `(S, L, D, F)_2`): [data_name_a, data_name_b, ...,],
            ...,
            group_name_x(data_type_x, such as `(S, D, F)): [data_name_q, data_name_w, ...,]
        }
    :param data_type: the type of data in data_dict, you have only TWO choices now:
        - `numpy` for np.ndarray data output
        - `xarray` for xr.DataArray data output

    return:
        - the grouped_data_dict, a dict of grouped data, format should be:
            {
                "date_0":{group_name_1: array_0, group_name_2: array_2, ... ,},
                "date_1":{group_name_1: array_0, group_name_2: array_2, ... ,},
                ...,
                "date_n":{group_name_1: array_0, group_name_2: array_2, ... ,}
            }

    """

    # ---- Step 1. Construct the empty grouped_data_dict ---- #
    grouped_data_dict = DataDict()

    # ---- Step 2. For loop all date and group the feature ---- #
    for date in data_dict.keys():
        for group_name, data_names in group_config.items():
            grouped_array = []
            for data_name in data_names:
                grouped_array.append(data_dict[date][data_name])
            if data_type == "numpy":
                grouped_data_dict[date][group_name] = np.concatenate(grouped_array, -1)  # transfer
            elif data_type == "xarray":
                grouped_data_dict[date][group_name] = xr.concat(grouped_array, dim="F")
            else:
                ValueError(data_type)
    return grouped_data_dict
