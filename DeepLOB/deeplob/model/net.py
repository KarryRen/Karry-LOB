# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/22 15:18

""" The deep-lob network. DeepLOBNet: for one code. """

from typing import List, Dict, Union
import torch
from torch import nn

from .modules import VoidModule
from .modules import get_conv2d_instance
from .modules import get_norm_instance


class DeepLOBNet(nn.Module):
    """ DeepLOB Network for single code.

    There have 4 parts:
        - Part 1. `Norm` for feature normalization
        - Part 2. `Feature Encoder` for feature encoding
        - Part 3. `Seq Encoder` for sequence feature encoding
        - Part 4. `Fully Connected Layers` for final prediction

    """

    def __init__(
            self,
            device: torch.device,
            norm_params: Dict[str, Union[list, dict]] = None,
            feature_encoder_params: Dict[str, dict] = None,
            fusion_index_list: List[List[int]] = None,
            seq_encoder_params: Dict[str, dict] = None,
            label_len: int = 1,
            **kwargs
    ):
        """ Initialization of the deep-lob net.

        :param device: the computing device
        :param norm_params: the params of normalization for each feature
            The format of this dict should be:
            {
                "norm_key_1": {"cls_kwargs": {"type": "type_1", "feature_list": []}, "init_kwargs": {}},
                "norm_key_2": {"cls_kwargs": {"type": "type_2", "feature_list": []}, "init_kwargs": {}},
                ...,
                "norm_key_n": {"cls_kwargs": {"type": "type_n", "feature_list": []}, "init_kwargs": {}}
            }
        :param feature_encoder_params: the params for each encoder of features
            The format of this dict should be:
            {
                "feature_type_1": {"param_name_1":param_value_1, "param_name_2":param_value_2, ...},
                "feature_type_2": {"param_name_1":param_value_1, "param_name_2":param_value_2, ...},
                ...
                "feature_type_n": {"param_name_1":param_value_1, "param_name_2":param_value_2, ...},
            }
        :param fusion_index_list: The fusion index list of encoded features, n feature_types will have n items !
        :param seq_encoder_params: the params for each encoder of each target feature type
            You have three choices: LSTM or GRU or RNN, the param should be basd on your choice.
            The format of this dict should be:
            {
                "LSTM/GRU/RNN":"param_name_1":param_value_1, "param_name_2":param_value_2, ...,
            }
        :param embedding_params: the embedding_params, working after cat
        :param label_len: the num of label length

        NOTE:
            - Here we first init Part 3 and 4, then init Part 1 and 2 for making sure param fixed
                (Seq Encoder & Fully Connected layers are same).

        """

        super(DeepLOBNet, self).__init__()

        self.device = device
        self.fusion_index_list = fusion_index_list
        self.LS = label_len

        # ---- Part 3. Seq Encoder (for sequence feature encoding) ---- #
        # get the seq encoder type based on the key
        seq_encoder_type = list(seq_encoder_params.keys())[0]
        # the input_size of seq encoder is the target fusion channel num
        self.fusion_channel_num = seq_encoder_params[seq_encoder_type]["input_size"]
        # construct the seq encoder based on the type
        if seq_encoder_type == "LSTM":
            self.seq_encoder = nn.LSTM(**seq_encoder_params["LSTM"]).to(device=device)
        elif seq_encoder_type == "GRU":
            self.seq_encoder = nn.GRU(**seq_encoder_params["GRU"]).to(device=device)
        elif seq_encoder_type == "RNN":
            self.seq_encoder = nn.RNN(**seq_encoder_params["RNN"]).to(device=device)
        else:
            raise TypeError(seq_encoder_type)

        # ---- Part 4. Fully Connected layers (for final prediction) ---- #
        self.fc = nn.Linear(seq_encoder_params[seq_encoder_type]["hidden_size"], 1, bias=False).to(device=device)

        # ---- Part 2. Feature Encoder (for feature encoding) ---- #
        self.feature_encoder_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward will have wrong !
        for fe_key in feature_encoder_params.keys():  # for loop the norm_params
            fe_cls_kwargs = feature_encoder_params[fe_key]["cls_kwargs"]  # get the kwargs of init class
            fe_init_kwargs = feature_encoder_params[fe_key]["init_kwargs"]  # get the kwargs of the instance
            self.feature_encoder_dict[fe_key] = get_conv2d_instance(cls_kwargs=fe_cls_kwargs, init_kwargs=fe_init_kwargs)  # get the instance
        # ensure all feature encoder modules are at device
        for fe_key, feature_encoder in self.feature_encoder_dict.items():
            self.feature_encoder_dict[fe_key] = feature_encoder.to(device=device)

        # ---- Part 1. Norm (for feature normalization) ---- #
        self.norm_dict = nn.ModuleDict({})  # must use nn.ModuleDict or backward will have wrong !
        for norm_key in norm_params.keys():  # for loop the norm_params
            if "cls_kwargs" in norm_params[norm_key].keys() and "init_kwargs" in norm_params[norm_key].keys():
                norm_cls_kwargs = norm_params[norm_key]["cls_kwargs"]  # get the kwargs of init norm class
                norm_init_kwargs = norm_params[norm_key]["init_kwargs"]  # get the kwargs of the instance
                self.norm_dict[norm_key] = get_norm_instance(cls_kwargs=norm_cls_kwargs, init_kwargs=norm_init_kwargs)  # get the instance
        # ensure all norm modules are at device, and is child of VoidModule class
        for norm_key, norm in self.norm_dict.items():
            norm = norm.to(device=device)
            assert isinstance(norm, VoidModule), f"The norm encoder of {norm_key} type ERROR !"

        # ---- Check the fusion list ---- #
        assert len(self.fusion_index_list) == len(self.feature_encoder_dict), \
            "Please check the fusion fusion_index_list ! If you have `n` features, you should have `n` fusion_index !"

    def forward(self, lob_features: dict):
        """ Forward computing of deep-lob net

        :param lob_features: the input feature dict.

        :return: predict result, shape=(bs, label_len, 1)

        """

        # ---- Step 0. To device ---- #
        normed_features_dict = {}
        for key in lob_features.keys():
            if key == "metadata":
                pass
            else:  # the ture feature
                bs_num = lob_features[key].shape[0]  # batch size
                seq_num = lob_features[key].shape[2]  # time steps
                # change to LIST (really important)
                normed_features_dict[key] = [lob_features[key]]

        # ---- Step 1. Create Fuse Empty Memory (all zeros) ---- #
        x_shape = (bs_num, self.fusion_channel_num, seq_num, 1)  # (bs, channel_num, time_steps, 1)
        x = torch.zeros(x_shape, device=self.device)

        # ---- Step 2. Read Data and do Normalization ---- #
        for key, norm_module in self.norm_dict.items():
            norm_module(normed_features_dict)

        # ---- Step 3. Use Encoder-Dict to do Feature Encoding and Fuse all features --- #
        fusion_index = 0
        for feature_type in self.feature_encoder_dict.keys():  # only encoding the last item
            x[:, self.fusion_index_list[fusion_index]] += self.feature_encoder_dict[feature_type](normed_features_dict[feature_type][-1])
            fusion_index += 1

        # ---- Step 4. Seq Encoding ---- #
        # transpose from (bs, c, time_steps, 1) to (bs, time_steps, c, 1)
        x = x.permute(0, 2, 1, 3)
        # cut off the last dim to (bs, time_steps, c)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        # do the lstm and get the output of each time step
        # the init state DEFAULT be ALL zeros, no matter the type of seq_encoder
        x, _ = self.seq_encoder(x)  # shape=(bs, time_steps, hidden_size)
        # get the output of the last label_len steps
        x = x[:, -self.LS:, :]  # shape=(bs, label_len, lstm_hidden_size)

        # ---- Step 4. Fully Connected ---- #
        # fully connected and get the final output, shape=(bs, label_len, 1)
        y_pred = self.fc(x)
        return {"label": y_pred}
