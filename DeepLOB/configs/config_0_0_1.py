# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19


""" Config file.

The config to facilitate factorial testing.

"""

import os
from typing import Dict, Union, List
import torch

from DeepLOB.task_util import environ_config
from DeepLOB.deeplob.utils import get_best_state_dict_path
from DeepLOB.deeplob.utils import load_data_dict

dict_templates = {
    "OnlyLOB":
        """
            anchors:
                needed_features: &needed_features
                    - - {file_name: LOB, transpose: [T, L, D, F]}

            0.baseline:
                needed_features: *needed_features
        """
}


# ---- Calculate the total channel ---- #
def cal_total_channel(conv_encoder_params: Dict, fusion_way: Union[str, List[str]]):
    """ Compute the total fusion channels.

    :return:
        - channel_end: the total channel number, the end channel idx + 1
        - fusion_index: the list of list [channel idx], n features will have n params

    """

    channel_end = 0
    fusion_index_list = []

    # Transfer the `str` fusion way to list
    if isinstance(fusion_way, str):
        # duplicate conv_encoder_params key number times
        fusion_way = [fusion_way for _ in conv_encoder_params]
        # cut off the first item
        fusion_way = fusion_way[1:]

    # Calculate the total channel
    if isinstance(fusion_way, list):
        keys = list(conv_encoder_params.keys())
        assert len(fusion_way) == len(keys) - 1, "FUSION_LIST is set WRONGLY !"
        # k = 0, the first fusion item
        channel_num = conv_encoder_params[keys[0]]["init_kwargs"]["conv_mid_channels"][-1]
        channel_end += channel_num
        fusion_index_list.append(list(range(channel_end - channel_num, channel_end)))
        # k > 0, the following fusion item
        for k in range(0, len(keys) - 1):
            channel_num = conv_encoder_params[keys[k + 1]]["init_kwargs"]["conv_mid_channels"][-1]
            if fusion_way[k] == "Add":  # Add in the end
                channel_start = channel_end - channel_num
            elif fusion_way[k] == "Cat":  # Cat from the end
                channel_start = channel_end
                channel_end += channel_num
            elif fusion_way[k].startswith("Add@"):  # Add from the @ channel
                channel_start = int(fusion_way[k].split("@")[-1])
                if channel_start + channel_num > channel_end:
                    channel_end = channel_start + channel_num
            else:
                raise ValueError(fusion_way[k])
            fusion_index_list.append(list(range(channel_start, channel_start + channel_num)))
    else:
        raise ValueError(fusion_way)

    return channel_end, fusion_index_list


class Config:
    """ The core class of config_x_y_z.py. """

    def __init__(self, config_yaml: dict, job_name: str):
        """ Init of config.

        :param config_yaml: the config yaml dict for one job, the format should be:
            { "needed_features": {...}, "dim_list": {...} ...,}, from `load_config()` interface in `__init__.py` file
        :param job_name: the name of this job

        """

        # ---- Step 1. Do the const init ---- #
        self.const_init()

        # ---- Step 2. Render the config by config_yaml ---- #
        self.render(config_yaml)
        self.job_name = job_name

        # ---- Step 3. Do the dynamic init ---- #
        self.dynamic_init()

    def const_init(self) -> None:
        """ Do the const init. Define some basic params."""

        # ---- Get the random seed dict ---- #
        self.SEED_DICT = {
            "Model": environ_config.get("seed_model"),  # random seed for model construction, control the random param init
            "DataLoader": environ_config.get("seed_dataloader"),  # random seed for DataLoader, control the dataloader init
            "Train": environ_config.get("seed_train")  # random seed for train, control the random for optimizing
        }

        # ---- Get the redo or not ---- #
        self.REDO = environ_config["redo"]  # control whether to need train or test

        # ---- Get the device ---- #
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Basic config of data ---- #
        self.DATA_ROOT_DICT = {"/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB/TensorEngineering": "TensorEngineering"}
        self.FUTURE_TYPE = environ_config["future_type"]
        self.TRAIN_DATES = environ_config["train_dates"]
        self.VALID_DATES = environ_config["valid_dates"]
        self.TEST_DATES = environ_config["test_dates"]
        self.FEATURE_LEN, self.LABEL_LEN = 5, 1
        self.TICK_NUM, self.START_TICK, self.END_TICK = 28800, 241, 28560
        self.NEEDED_LABELS = self.NEEDED_LABELS = {"Label": {"Label": {"file_name": "HFTLabel", "slice": [1], "scale": 10000, "shift": (-1, 0.0)}}}

        # ---- Detail config of features ----- #
        self.NEEDED_FEATURES = {}
        self.NEEDED_FEATURES_DATA = []
        self.NEEDED_FEATURES_SHAPE = {}
        self.NEEDED_FEATURES_DIMS = {}  # for shape check
        self.PAF_SHIFT_K = {}

        # ---- The dataloader config ---- #
        self.BATCH_SIZE = 8196
        self.TRAIN_DATASET_TYPE = "DeepLOBDataset_DailySample"
        if self.TRAIN_DATASET_TYPE == "DeepLOBDataset_DailySample":
            self.TICK_NUM_IN_ONE_SAMPLE = 10
            self.TRAIN_BATCH_SIZE = self.BATCH_SIZE // (self.TICK_NUM_IN_ONE_SAMPLE * len(self.TRAIN_DATES))

        # ---- Model construction ---- #
        # - the params of normalization
        self.NORM_PARAMS = {}
        # - the params of feature encoder
        self.FEATURE_ENCODER_PARAMS = {}
        # - the params for fusion
        # you have only 3 choices now, "Cat", "Add" or "Add@"
        # if you set the fusion way list as a str, that means the fusion way will be used for all feature
        # else you should set the clear fusion way corresponding to the feature encoder keys !!!
        self.FUSION_WAY_LIST = "Cat"
        # - pretrain path of model
        self.PRETRAIN = None

        # ---- Optimizer ---- #
        self.EPOCHS = 10
        self.OPTIM_PARAMS = {
            "cls_kwargs": {"name": "AdamW"},
            "init_kwargs": {"lr": 0.010, "betas": (0.99, 0.999), "weight_decay": 0.1},
            "freeze": []
        }

        # ---- Main metric using to select models ---- #
        self.MAIN_METRIC = "valid_rescaled_daily_mean_r2"

        # ---- PATH ---- #
        self.SEED_STR = ';'.join([f"{k}:{v}" for k, v in self.SEED_DICT.items()])
        self.EXPERIMENTS_PATH = "Save"

        # ---- Compatibility configs ---- #
        self.DICT = {
            "use_sample_std_weight": False,
            "loss": {"MTL": {"reduction": "sum", "ones_weight": False}},
            "use_valid_beta": False
        }

    def render(self, config_yaml: dict) -> None:
        """ Render the config by config_yaml. """

        # ---- Render the features config to `NEEDED_FEATURES_DATA` ---- #
        # - extend the baseline features
        if "needed_features" in config_yaml.keys():
            self.NEEDED_FEATURES_DATA.extend(config_yaml["needed_features"])
        # - extend the adding features
        if "adding_features" in config_yaml.keys():
            self.NEEDED_FEATURES_DATA.extend(config_yaml["adding_features"])

        # ---- Render the `pretrain` config ---- #
        if "pretrain" in config_yaml.keys():
            # read the yaml get the `job_name` and `patterns`
            if "job_name" in config_yaml["pretrain"].keys():
                job_name = config_yaml["pretrain"]["job_name"]
                patterns = config_yaml["pretrain"]["patterns"]
                # get the pretrain model path
                pretrain_root_path = f"{self.EXPERIMENTS_PATH}/{job_name}"
                # set `pretrain` dict
                self.PRETRAIN = {
                    "patterns": patterns,
                    "path": get_best_state_dict_path(
                        f"{pretrain_root_path}/{self.FUTURE_TYPE}/model_date_{self.TEST_DATES[0]}/rs_{self.SEED_STR}/trained_models/",
                        self.MAIN_METRIC
                    )
                }
            else:
                raise "If you want to use `pretrain`, you need set the pretrained model `job_name` in yaml !!"
            # set the `freeze`
            if "freeze" in config_yaml["pretrain"].keys():
                freeze = config_yaml["pretrain"]["freeze"]
                assert set(freeze) <= set(patterns), f"The freeze module set {set(freeze)} must smaller than patten module set {set(patterns)} !!"
                self.OPTIM_PARAMS["freeze"] = freeze

    def dynamic_init(self):
        """ Dynamic init. """

        # ---- Make the `NEEDED_FEATURES` by `NEEDED_FEATURES_DATA` ---- #
        feature_idx = 0
        for data_array_idx, data_array_config in enumerate(self.NEEDED_FEATURES_DATA):
            self.NEEDED_FEATURES[f"DataArray_{data_array_idx}"] = {}
            for feature_config in data_array_config:
                self.NEEDED_FEATURES[f"DataArray_{data_array_idx}"][f"X_{feature_idx}"] = feature_config
                feature_idx += 1

        # ---- Read one date feature to get the DIM and SHAPE ---- #
        one_dates = self.VALID_DATES[-1:]
        data_dict = load_data_dict(one_dates, self.FUTURE_TYPE, self.NEEDED_FEATURES, self.DATA_ROOT_DICT, data_type="xarray")
        for data_array_key in self.NEEDED_FEATURES.keys():
            # get the dim and shape of feature
            feature_dim = list(data_dict[one_dates[0]][data_array_key].dims)
            feature_shape = list(data_dict[one_dates[0]][data_array_key].shape)
            # check the dim
            assert feature_dim[0] == "T", f"The first dim of feature MUST be `T`, but {data_array_key} is {feature_dim[0]} !"
            # save dim and shape for future using
            self.NEEDED_FEATURES_DIMS[data_array_key] = feature_dim[1:]
            self.NEEDED_FEATURES_SHAPE[data_array_key] = feature_shape[1:]

        # ---- Set and check the NORM and FEATURE_ENCODER params ---- #
        # set params
        for feature_name, feature_config in self.NEEDED_FEATURES.items():
            shape_list = self.NEEDED_FEATURES_SHAPE[feature_name]
            # for FEATURE_ENCODER_PARAMS
            # - init the conv_mid_channels based on shape list
            if len(shape_list) == 1:
                conv_mid_channels = [1, 4]
            elif len(shape_list) == 2:
                conv_mid_channels = [1, 8, 4]
            elif len(shape_list) == 3:
                conv_mid_channels = [1, 16, 16, 16]
            elif len(shape_list) == 4:  # PAF or TAF or SAF, one dim is `hf` not used for net construction
                conv_mid_channels = [1, 16, 16, 2]
                assert self.NEEDED_FEATURES_DIMS[feature_name][0] == "HF", "AF dim ERROR. First dim must be `HF` !"
                shape_list = shape_list[1:]  # change the shape list
            else:
                raise ValueError(shape_list)
            # - init the encoder params
            this_encoder_params = {
                "cls_kwargs": {"type": "Conv2d_%dDim" % len(shape_list)},
                "init_kwargs": {
                    "conv_mid_channels": conv_mid_channels, "dim_list": shape_list,
                    "bn_momentum": 0.5, "bn_type": "BN"
                }
            }
            # - set the encoder params
            self.FEATURE_ENCODER_PARAMS[feature_name] = this_encoder_params
            # for NORM PARAMS (here we use the length of shape list to judge the feature type)
            # - have 1 or 2 dim, then need BN Norm
            if len(shape_list) < 3:
                norm_key = f"BN_{len(shape_list)}Dim_{feature_name}"
                this_norm_params = {
                    "cls_kwargs": {"type": f"BN_{len(shape_list)}Dim"},
                    "init_kwargs": {"name_list": [feature_name], "bn_type": "BN", "feature_dim": shape_list[-1]}
                }
                self.NORM_PARAMS[norm_key] = this_norm_params
            # - have 3 dims, then might need `PVNorm` and `PVFillna`
            elif len(shape_list) == 3:
                assert len(feature_config) == 1, "Now only support 1 (S, L, D, F) feature"
                file_name = next(iter(feature_config.values()))["file_name"]  # get the feature name
                # PV Norm for LOB, PAF or TAF or SAF, other features will just use BN
                if file_name.startswith("LOB") or file_name.startswith("PAF") or file_name.startswith("TAF") or file_name.startswith("SAF"):
                    if "PVNorm" not in self.NORM_PARAMS.keys():  # init the PVNorm
                        self.NORM_PARAMS["PVNorm"] = {
                            "cls_kwargs": {"type": "PVNorm"},
                            "init_kwargs": {"name_list": [feature_name], "mean_std_time_steps": self.FEATURE_LEN}
                        }
                    else:
                        self.NORM_PARAMS["PVNorm"]["init_kwargs"]["name_list"].append(feature_name)
                else:
                    norm_key = f"BN_{len(shape_list)}Dim_{feature_name}"
                    this_norm_params = {
                        "cls_kwargs": {"type": f"BN_{len(shape_list)}Dim"},
                        "init_kwargs": {"name_list": [feature_name], "bn_type": "BN", "feature_dim": shape_list[-1]}
                    }
                    self.NORM_PARAMS[norm_key] = this_norm_params
                # PV Fillna
                if file_name.startswith("PAF") or file_name.startswith("TAF") or file_name.startswith("SAF"):
                    self.PAF_SHIFT_K[feature_name] = 0
                    if "PVFillna" not in self.NORM_PARAMS.keys():  # init the PVFillna
                        self.NORM_PARAMS["PVFillna"] = {
                            "cls_kwargs": {"type": "PVFillna"},
                            "init_kwargs": {"name_list": [feature_name], "fill_value": 0.0}}
                    else:
                        self.NORM_PARAMS["PVFillna"]["init_kwargs"]["name_list"].append(feature_name)
            # - other dims are wrong
            else:
                raise ValueError(shape_list)
        # check and adjust the params based on some principles
        # - principle 1. in PV Norm the `LOB` must be first
        pv_norm_first_name = self.NORM_PARAMS["PVNorm"]["init_kwargs"]["name_list"][0]  # get the first name
        assert self.NEEDED_FEATURES[pv_norm_first_name][next(iter(self.NEEDED_FEATURES[pv_norm_first_name]))]["file_name"].startswith("LOB"), \
            "The sequence of PV Norm name_list is not right !"
        # - principle 2. PVFillna must be the first key
        if "PVFillna" in self.NORM_PARAMS.keys():
            # -- get the PVFillna
            norm_param_copy = {"PVFillna": self.NORM_PARAMS["PVFillna"]}
            self.NORM_PARAMS.pop("PVFillna")  # pop it
            # -- get the other norm param
            for key, item in self.NORM_PARAMS.items():
                norm_param_copy[key] = item
            # -- set back
            self.NORM_PARAMS = norm_param_copy

        # ---- Set the path ---- #
        self.SAVE_PATH = f"{self.EXPERIMENTS_PATH}/{self.job_name}/{self.FUTURE_TYPE}/model_date_{self.TEST_DATES[0]}/rs_{self.SEED_STR}/"
        self.LOG_FILE = self.SAVE_PATH + "deeplob_log.log"
        self.MODEL_SAVE_PATH = self.SAVE_PATH + "trained_models/"
        self.TOTAL_CHANNEL, self.FUSION_INDEX_LIST = cal_total_channel(self.FEATURE_ENCODER_PARAMS, self.FUSION_WAY_LIST)
        print(f"TOTAL_CHANNEL_NUM: {self.TOTAL_CHANNEL}, FUSION_INDEX_LIST: {self.FUSION_INDEX_LIST}")

        # ---- Build the sequence feature encoder ---- #
        self.SEQ_ENCODER_PARAMS = {
            "LSTM": {
                "input_size": self.TOTAL_CHANNEL,
                "hidden_size": 32,
                "num_layers": 1,
                "batch_first": True
            }
        }

        # ---- Summary of NET PARAMS ---- #
        self.NET = {
            "norm_params": self.NORM_PARAMS,
            "feature_encoder_params": self.FEATURE_ENCODER_PARAMS,
            "fusion_index_list": self.FUSION_INDEX_LIST,
            "seq_encoder_params": self.SEQ_ENCODER_PARAMS,
            "label_len": self.LABEL_LEN,
            "pretrain": self.PRETRAIN
        }
