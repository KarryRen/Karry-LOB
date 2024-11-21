# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 19:19

""" Config file.
    On the one hand, parsing the relevant configuration in `environ_config` to construct hyperparameters;
    On the other hand, creating some new hyperparameters.

Version 0.0.1 (release on 20241121 by Karry Ren).
This is the baseline.

"""

import os
from typing import Dict, Union, List

from DeepLOB.task_util import environ_config
from DeepLOB.deeplob.utils import get_best_state_dict_path

dict_templates = {
    "OnlyLOB":
        """
            LOB: {}
        """
}


# ---- Calculate the total channel ---- #
def cal_total_channel(conv_encoder_params: Dict, fusion_way: Union[str, List[str]]):
    """ Compute the total fusion channels.

    return:
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
    """ The core class of config_x_y_z.py """

    def __init__(self, config_yaml: dict, job_name: str):
        """ Init of config.

        :param config_yaml: the config yaml dict for one job, the format should be:
            {
                "needed_features": {...},
                "dim_list": {...},
                ...,
            }
        :param job_name: the name of this job

        """

        self.job_name = job_name

        # ---- Step 1. Do the CONST init ---- #
        self.const_init()

        # ---- Step 2. Render by the config yaml ---- #
        self.render(config_yaml)

        # ---- Step 3. Do the dynamic init ---- #
        self.dynamic_init()

    def const_init(self):
        """ Do the const init. """

        # ---- Init the random seed dict ---- #
        self.SEED_DICT = {
            "Model": environ_config.get("seed_model"),  # random seed for model construction, control the random param init
            "DataLoader": environ_config.get("seed_dataloader"),  # random seed for DataLoader, control the dataloader init
            "Train": environ_config.get("seed_train")  # random seed for train, control the random for optimizing
        }

        # ---- Init the redo ---- #
        self.REDO = environ_config["redo"]  # control whether to need train or test

        # ---- Init the params about the dataset ---- #
        self.DATA_ROOT_DICT = {"/Users/karry/KarryRen/Fintech/Quant/Projects/Data/Karry-LOB": "TensorEngineering"}
        self.FUTURE_TYPE = environ_config["future_type"]
        # dates
        self.TRAIN_DATES = environ_config["train_dates"]
        self.VALID_DATES = environ_config["valid_dates"]
        self.TEST_DATES = environ_config["test_dates"]
        # length
        self.FEATURE_LEN, self.LABEL_LEN = 5, 1
        # tick number (begin from 1)
        self.TICK_NUM, self.START_TICK, self.END_TICK = 28800, 241, 28560  # the total tick_num of one day, the start and end tick
        # all needed features (defaulted)
        self.NEEDED_FEATURES = {"(S, L, D, F)": {"LOB": {"transpose": ("T", "L", "D", "F")}}}
        self.PAF_SHIFT_K = {}
        # all needed labels
        self.NEEDED_LABELS = {"(S, F)_Label": {"HFTLabel": {"slice": [1], "scale": 10000, "shift": (-1, 0.0)}}}

        # ---- Init the params about dataloader ---- #
        self.BATCH_SIZE = 8196
        self.TRAIN_DATASET_TYPE = "DeepLOBDataset_DailySample"
        if self.TRAIN_DATASET_TYPE == "DeepLOBDataset_DailySample":
            self.TICK_NUM_IN_ONE_SAMPLE = 10
            self.TRAIN_BATCH_SIZE = self.BATCH_SIZE // (self.TICK_NUM_IN_ONE_SAMPLE * len(self.TRAIN_DATES))

        # ---- The basic params of model construction ---- #
        # - the params of normalization, the sequence of keys are really important, `PVFillna` must in front of `PVNorm` !!
        self.NORM_PARAMS = {
            "PVFillna": {},
            "PVNorm": {  # only use the first one in `name_list` to compute mean&std !!
                "cls_kwargs": {"type": "PVNorm"},
                "init_kwargs": {"name_list": ["(S, L, D, F)"], "mean_std_time_steps": self.FEATURE_LEN}
            }
        }
        # - the params of feature encoder
        self.FEATURE_ENCODER_PARAMS = {
            "(S, L, D, F)": {
                "cls_kwargs": {"type": "Conv2d_3Dim"},
                "init_kwargs": {"conv_mid_channels": [1, 16, 16, 16], "dim_list": [5, 2, 2], "bn_momentum": 0.5, "bn_type": "BN"}
            }
        }
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

        # ---- Save path ---- #
        self.SEED_STR = ';'.join([f"{k}:{v}" for k, v in self.SEED_DICT.items()])

        # ---- Compatibility params ---- #
        self.DICT = {
            "use_sample_std_weight": False,
            "loss": {"MTL": {"reduction": "sum", "ones_weight": False}},
            "use_valid_beta": False
        }

    def render(self, config_yaml: dict):
        """ Use the config yaml to render the config. """

        if len(config_yaml.keys()) > 0:
            # the config of `need_features`
            assert "needed_features" in config_yaml.keys(), "`needed_features` must in yaml"
            needed_features_config = config_yaml["needed_features"]
            # the config of `conv_encoder`
            assert "conv_encoder" in config_yaml.keys(), "`conv_encoder` must in yaml"
            conv_encoder_config = config_yaml["conv_encoder"]
            # check the feature type
            assert list(needed_features_config.keys()) == list(
                conv_encoder_config.keys()), "feature type of `needed_features` and `conv_encoder` are not same !"
            feature_type_list = list(conv_encoder_config.keys())
            # update the feature
            self.NEEDED_FEATURES.update(needed_features_config)
            # update the net structure
            # - norm
            if "norm" in config_yaml.keys():
                norm_config = config_yaml["norm"]
                for norm_key in norm_config.keys():
                    self.NORM_PARAMS[norm_key] = {
                        "cls_kwargs": {"type": norm_config[norm_key]["type"]},
                        "init_kwargs": {"name_list": norm_config[norm_key]["name_list"], "bn_type": "BN",
                                        "feature_dim": norm_config[norm_key]["feature_dim"]}
                    }
            # - conv encoder
            for feature_type in feature_type_list:
                conv_mid_channels = conv_encoder_config[feature_type]["conv_mid_channels"]
                dim_list = conv_encoder_config[feature_type]["dim_list"]
                self.FEATURE_ENCODER_PARAMS[feature_type] = {
                    "cls_kwargs": {"type": f"Conv2d_{len(dim_list)}Dim"},
                    "init_kwargs": {"conv_mid_channels": conv_mid_channels, "dim_list": dim_list, "bn_momentum": 0.5, "bn_type": "BN"}
                }
        # - pretrain
        if "pretrain" in config_yaml.keys():
            self.PRETRAIN = {
                "patterns": config_yaml['pretrain']['patterns'],
                "path": get_best_state_dict_path(
                    f"{config_yaml['pretrain']['path']}/{self.FUTURE_TYPE}/model_date_{self.VALID_DATES[-1]}"
                    f"/rs_{self.SEED_STR}/trained_models", self.MAIN_METRIC
                )}
        # - freeze
        if "freeze" in config_yaml.keys():
            self.OPTIM_PARAMS["freeze"] = config_yaml["freeze"]

    def dynamic_init(self):
        """ Dynamic init some params. """

        self.SAVE_PATH = (
            f"save/{self.job_name}/{self.FUTURE_TYPE}/model_date_{self.VALID_DATES[-1]}/rs_{self.SEED_STR}/"
        )
        self.LOG_FILE = self.SAVE_PATH + "deeplob_log.log"
        self.MODEL_SAVE_PATH = self.SAVE_PATH + "trained_models/"
        self.TOTAL_CHANNEL, self.FUSION_INDEX_LIST = cal_total_channel(self.FEATURE_ENCODER_PARAMS, self.FUSION_WAY_LIST)

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
            "label_len": self.LABEL_LEN,
            "pretrain": self.PRETRAIN,
            "norm_params": self.NORM_PARAMS,
            "feature_encoder_params": self.FEATURE_ENCODER_PARAMS,
            "fusion_index_list": self.FUSION_INDEX_LIST,
            "seq_encoder_params": self.SEQ_ENCODER_PARAMS
        }
