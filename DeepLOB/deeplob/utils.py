# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" Some util functions. """

import re
import random
import os
from typing import Dict, List, Union
import h5py
import numpy as np
import torch
import pandas as pd
import logging

from DeepLOB.deeplob.datadict.deeplob_datadict import gen_data_dict_config, gen_data_dict, group_data_dict, DataDict


def fix_random_seed(seed: int) -> None:
    """ Fix the random seed to decrease the random of training.
    Ensure the reproducibility of the experiment.

    :param seed: the random seed number

    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def analyze_running_bool(config, redo: bool, checkpoints: List[str]) -> Dict[str, bool]:
    """ Analyze the running bool based on the file existing.

    :param config: the config file
    :param redo: is redo or not, if redo just Train&Test directly (Train&Test the new models)
    :param checkpoints: the list to check which determine whether to redo

    return:
        - running_bool: the running bool dict means whether to need Train or Test

    """

    # ---- Init the running_bool (default is redo) ---- #
    running_bool = {"Train": True, "Test": True}

    # ---- Based on the `redo` to set the `running_bool` ---- #
    if redo:  # redo, just Train & Test directly
        pass
    else:  # not redo, then set the running_bool based on the situation
        # whether Train, based on having the last epoch trained model or not
        model_path = config.MODEL_SAVE_PATH + f"model_statedict_epoch_{config.EPOCHS - 1}.pkl"
        running_bool["Train"] = (not os.path.exists(model_path))
        # whether Test, based on having the label prediction result or not
        if not running_bool["Train"]:  # if Train then must Test, if not Train should consider
            bool_test = False
            for file in checkpoints:  # check all files in checkpoints, if 1 not existed, then need test
                file_full_path = config.SAVE_PATH + file
                if os.path.exists(file_full_path):
                    if file in ["valid_label_pred.h5", "test_label_pred.h5"]:
                        feature_dict = {"dates": None}
                        with h5py.File(file_full_path, "r") as fp:
                            for key in feature_dict.keys():
                                array = fp[key][:]
                                if not np.issubdtype(array.dtype, np.number):
                                    array = array.astype(np.str_)  # special operation for `str`
                                feature_dict[key] = array
                        if file == "valid_label_pred.h5" and (set(feature_dict["dates"]) < set(config.VALID_DATES)):
                            bool_test = True
                        elif file == "test_label_pred.h5" and (set(feature_dict["dates"]) < set(config.TEST_DATES)):
                            bool_test = True
                else:
                    bool_test = True
            running_bool["Test"] = bool_test
    return running_bool


def load_data_dict(
        dates: List[str], codes: Union[str, List[str]], needed_data: Dict,
        data_root_path_dict: Dict[str, str], verbose: bool = False, data_type: str = "numpy"
) -> DataDict:
    """ Load data dict to the memory.

    :param dates: the dates to load data dict
    :param codes: the codes to load data dict
    :param needed_data: the needed data
    :param data_root_path_dict: the root path dict to load data
    :param verbose: a boolean flag indicating whether to enable verbose output.
    :param data_type: numpy or xarray

    """

    # ---- Step 1. Generate the config for data dict ---- #
    data_dict_config = gen_data_dict_config(needed_data=needed_data)

    # ---- Step 2. Generate the data dict based on the config ---- #
    data_dict = gen_data_dict(data_root_path_dict=data_root_path_dict, codes=codes, dates=dates, data_dict_config=data_dict_config, verbose=verbose)

    # ---- Step 3. Group data_dict as data_dict_grouped ---- #
    # init the empty config dict
    group_config = {}
    # construct the group_config based on the `NEEDED_LABELS` and `NEEDED_FEATURES` dict
    # the key is data_type
    for key, value in needed_data.items():
        group_config[key] = list(value.keys())  # [name_1, name_2, ..., name_n]
    # group the data dict based on the grou_name which is data_type
    grouped_data_dict = group_data_dict(data_dict, group_config, data_type)
    return grouped_data_dict


def get_best_state_dict_path(model_save_path: str, metric: str) -> str:
    """ Using the metric to select the best model after training.

    :param model_save_path: the path of saving models
    :param metric: the dependent metric to select best model

    return:
        - statedict_path : the path of best model statedict

    """

    # ---- Step 1. Read the metric df and test the `metric` ---- #
    metric_df = pd.read_csv(f"{model_save_path}/model_pytorch_metric.csv", index_col=0)

    # ---- Step 2. Get the path of best epoch model ---- #
    best_epoch = metric_df.index[np.argmax(metric_df[metric].values)]
    statedict_path = model_save_path + f"model_statedict_epoch_{best_epoch}.pkl"
    return statedict_path


def isin_patterns(name: str, patterns: List[str]) -> bool:
    """ Check whether name in patterns. """

    bool_isin = False
    for pattern in patterns:
        if re.search(pattern, name):
            bool_isin = True
            break
    return bool_isin


def load_patterns_state_dict(model: torch.nn.Module, pretrain_model_state_dict: Dict[str, torch.Tensor], patterns: List[str]) -> str:
    """ Load the patterns state dict from `pretrain_model_state_dict` to `model`.

    :param model: the target model
    :param pretrain_model_state_dict: the model state dict
    :param patterns: the param patterns

    :return: info

    """

    try:
        for key in model.state_dict().keys():  # for loop the `state_dict` of model
            if key in pretrain_model_state_dict:
                if isin_patterns(key, patterns):
                    if "seq_encoder" not in key:
                        assert model.state_dict()[key].shape == pretrain_model_state_dict[key].shape, f"{key} must be same shape"
                        model.state_dict()[key].copy_(pretrain_model_state_dict[key])
                        logging.info(f"|| Param `{key}` is loaded. ||")
                    else:
                        model.state_dict()[key] = load_lstm_param(key.split(".")[-1], model.state_dict()[key], pretrain_model_state_dict[key])
                        logging.info(f"|| Param `{key}` is loaded by the seq_encoder interface. ||")
                else:
                    logging.info(f"|| Pretrain model param `{key}` is in not in patterns. Use init ||")
            else:
                logging.info(f"|| Pretrain model missing param `{key}`. Use init ||")
        return "<All patterns matched successfully>"
    except Exception as e:
        return str(e)


def load_lstm_param(param_key: str, target_param_tensor: torch.Tensor, pretrain_param_tensor: torch.Tensor) -> torch.Tensor:
    """ Load the lstm param from pretrain param.

    :param param_key: the key of param, MUST be the format in [`weight_ih_ln`, bias_ih_ln`, `weight_hh_ln`, bias_hh_ln`]
    :param target_param_tensor: the tensor of target param,
        - shape=(4 * hidden_size, target_input_size) for weight_ih_ln
        - shape=(4 * hidden_size, hidden_size) for weight_hh_ln
        - shape=(4 * hidden_size) for bias
    :param pretrain_param_tensor: the tensor of pretrain source param,
        - shape=(4 * hidden_size, pretrain_input_size) for weight_ih_ln
        - shape=(4 * hidden_size, hidden_size) for weight_hh_ln
        - shape=(4 * hidden_size) for bias

    :return:
        - target_param_tensor: the tensor of target param

    NOTE:
        - `target_input_size` and `pretrain_input_size` might be NOT SAME, but there must be a correspondence.
        - target_input_size MUST >= pretrain_input_size
        - Now, we only support the same hidden_size.
    """

    if "weight_hh" in param_key or "bias" in param_key:
        # directly set value
        target_param_tensor.copy_(pretrain_param_tensor)
    elif "weight_ih" in param_key:
        # get hidden_size
        assert target_param_tensor.shape[0] == pretrain_param_tensor.shape[0], "Now, we only support the same hidden_size."
        hidden_size = target_param_tensor.shape[0] // 4
        # get the input_size and check it
        target_input_size, pretrain_input_size = target_param_tensor.shape[1], pretrain_param_tensor.shape[1]
        assert target_input_size >= pretrain_input_size, "target_input_size MUST >= pretrain_input_size"
        # split param
        split_tpt = target_param_tensor.reshape(4, hidden_size, target_input_size)
        split_ppt = pretrain_param_tensor.reshape(4, hidden_size, pretrain_input_size)
        # set value
        split_tpt[:, :, :pretrain_input_size] = split_ppt
        # reshape back
        target_param_tensor.copy_(split_tpt.reshape(4 * hidden_size, target_input_size))
    else:
        raise TypeError(param_key)

    return target_param_tensor
