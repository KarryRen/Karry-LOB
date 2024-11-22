# -*- coding: utf-8 -*-
# @author : MaMing, RenKai (intern in HIGGS ASSET)
# @time   : 2023/11/25 12:54
#
# pylint: disable=no-member

""" The deep-lob network.
    - DeepLOBNet: for one code.
    - DeepLOBNetMultiCodes: for single code.

"""

from typing import List, Dict, Union
import torch
from torch import nn

from .modules import VoidModule
from .modules import get_conv2d_instance
from .modules import get_norm_instance
from .modules.batch_norm import get_bn_instance


class ConstLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, device, parameters=False):
        super(ConstLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_parameters = parameters
        self.const = torch.zeros(out_features, in_features).to(device=device)
        if parameters:
            self.weight = nn.Parameter(torch.rand(out_features, in_features).to(device=device))
        else:
            self.weight = None
            self.register_parameter("dummy", None)  # 需要添加这句，否则上层调用model.parameters()会有问题

    def forward(self, input):
        if self.use_parameters:
            output = torch.matmul(input, (self.const + self.weight).t())
        else:
            output = torch.matmul(input, self.const.t())
        return output

    def set_fusion_const(self, fusion_index_list):
        # init the weight by fusion_index_list
        in_index = 0
        out_index = 0
        for out_indexs in fusion_index_list:
            for out_index in out_indexs:
                self.const.data[out_index, in_index] = 1.0
                in_index += 1


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
            embedding_params=None,
            label_len: int = 1,
            use_label_bn: bool = False,
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
        self.use_label_bn = use_label_bn

        # ---- Label BN ---- #
        if self.use_label_bn:
            self.label_bn = get_bn_instance(bn_type="GBN", feature_dim=1, affine=False).to(device=device)
        else:
            self.label_bn = None

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

        # ---- TODO: No using now, will format in the future ---- #
        if embedding_params:
            # 不使用 nn.Linear 是因为它会自动调用随机初始化, 影响 init
            if embedding_params["requires_grad_"]:
                # self.embedding = nn.Linear(in_features=embedding_params["in_features"], out_features=embedding_params["out_features"],
                #                            bias=embedding_params["bias"]).to(device=device)
                self.embedding = ConstLinear(in_features=embedding_params["in_features"], out_features=embedding_params["out_features"],
                                             bias=embedding_params["bias"], device=device, parameters=True)
            else:
                self.embedding = ConstLinear(in_features=embedding_params["in_features"], out_features=embedding_params["out_features"],
                                             bias=embedding_params["bias"], device=device)
                assert (embedding_params["fusion_way_init"])
            if embedding_params["fusion_way_init"]:
                self.embedding.set_fusion_const(self.fusion_index_list)
        else:
            self.embedding = None

    def forward(self, lob_features: dict, labels: Union[dict, None] = None):
        """ Forward computing of deep-lob net

        :param lob_features: the input feature dict.
        :param labels: the labels

        :return: predict result, shape=(bs, label_len, 1)

        """

        if self.training:
            if self.use_label_bn:
                # label bn
                _ = self.label_bn(labels["label"].unsqueeze(-1))
        else:
            assert (labels is None)

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
        if self.embedding:
            x_shape = (bs_num, self.embedding.in_features, seq_num, 1)  # (bs, channel_num, time_steps, 1)
        else:
            x_shape = (bs_num, self.fusion_channel_num, seq_num, 1)  # (bs, channel_num, time_steps, 1)
        x = torch.zeros(x_shape, device=self.device)

        # ---- Step 2. Read Data and do Normalization ---- #
        for key, norm_module in self.norm_dict.items():
            norm_module(normed_features_dict)

        # ---- Step 3. Use Encoder-Dict to do Feature Encoding and Fuse all features --- #
        if self.embedding:
            # 矩阵乘法实现融合
            x = []
            for feature_type in self.feature_encoder_dict.keys():
                x.append(self.feature_encoder_dict[feature_type](normed_features_dict[feature_type][-1]))
            x = torch.concat(x, dim=1)
            x = self.embedding(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            # 用 fusion_index 区分 cat 和 add 两种融合方式
            fusion_index = 0
            for feature_type in self.feature_encoder_dict.keys():
                x[:, self.fusion_index_list[fusion_index]] += self.feature_encoder_dict[feature_type](
                    normed_features_dict[feature_type][-1]  # only encoding the last item
                )
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
        # use label bn, then adjust the y_pred
        if self.use_label_bn:
            y_pred = y_pred * torch.sqrt(self.label_bn.running_var)
        return {"label": y_pred}


class DeepLOBNetMultiCodes(nn.Module):
    def __init__(self, codes: List[str], **kwargs):
        """ 
        DeepLOBNet对code循环, 不共享任何参数。
        """

        super(DeepLOBNetMultiCodes, self).__init__()
        self.device = kwargs.get("device")
        self.codes = codes
        # ---- Part 1. init deeplobnet ---- #
        self.net_dict = nn.ModuleDict({})
        for k in range(len(self.codes)):
            self.net_dict[f"{k}"] = DeepLOBNet(**kwargs)

    def forward(self, lob_features: dict):
        """ Forward computing of deep-lob net

        :param lob_features: the input feature dict

        :return: predict result, shape=(bs, code, label_len, 1)

        """
        y_pred = []
        # 拆分数据
        for k in range(len(self.codes)):
            this_lob_features = {}
            for key in lob_features.keys():
                if key == "metadata":
                    this_lob_features[key] = lob_features[key]
                else:
                    this_lob_features[key] = lob_features[key][:, k:k + 1]  # bs * code * ...
            this_pred = self.net_dict[f"{k}"](this_lob_features)["label"]
            y_pred.append(this_pred)
        # 合并
        y_pred = torch.stack(y_pred, dim=1)  # bs * code * label num * label step
        return {"label": y_pred}


class DeepLOBNetMultiCodesShared(nn.Module):
    def __init__(self, codes: List[str],
                 norm_params: Dict[str, Union[list, dict]],
                 feature_encoder_params: dict,
                 fusion_index_list: List[List[int]],
                 seq_encoder_params: dict,
                 device: torch.device,
                 label_len: int = 1,
                 ):
        """ 
        共享卷积和Lstm模块，每个code拥有自己的norm和fc模块
        """

        super(DeepLOBNetMultiCodes, self).__init__()
        self.device = device
        self.fusion_index_list = fusion_index_list
        self.LS = label_len
        self.codes = codes
        # ---- Part 1. Seq Encoder. Code Share ---- #
        seq_encoder_type = list(seq_encoder_params.keys())[0]  # get the seq encoder type based on the key
        self.fusion_channel_num = seq_encoder_params[seq_encoder_type]["input_size"]
        if seq_encoder_type == "LSTM":
            self.seq_encoder = nn.LSTM(**seq_encoder_params["LSTM"]).to(device=device)
        elif seq_encoder_type == "GRU":
            self.seq_encoder = nn.GRU(**seq_encoder_params["GRU"]).to(device=device)
        elif seq_encoder_type == "RNN":
            self.seq_encoder = nn.RNN(**seq_encoder_params["RNN"]).to(device=device)
        else:
            raise TypeError(seq_encoder_type)

        # ---- Part 2. Fully Connected layers. Code Different ---- #
        self.fc_dict = nn.ModuleDict({})
        for k in range(len(self.codes)):
            self.fc_dict[f"{k}"] = nn.Linear(seq_encoder_params[seq_encoder_type]["hidden_size"], 1, bias=False).to(device=device)

        # ---- Part 3. Feature Encoder. Code Share ---- #
        self.feature_encoder_dict = nn.ModuleDict({})  # must use nn.Dict or backward wrong !
        for feature_type, params in feature_encoder_params.items():
            temp_class = conv2d_class_loader(class_type=feature_type.split("_")[0])
            self.feature_encoder_dict[feature_type] = temp_class(**params).to(device=device)
        # ---- Part 4. Norm Encoder. Code Split ---- #
        self.norm_dict = {}
        for k in range(len(self.codes)):
            self.norm_dict[k] = nn.ModuleDict({})
            for norm_key in norm_params.keys():  # for loop the norm_params
                norm_cls_kwargs = norm_params[norm_key]["cls_kwargs"]  # get the kwargs of init norm class
                norm_init_kwargs = norm_params[norm_key]["init_kwargs"]  # get the kwargs of the instance
                self.norm_dict[k][norm_key] = get_norm_instance(cls_kwargs=norm_cls_kwargs, init_kwargs=norm_init_kwargs)  # get the instance for norm
            # ensure norm dict at device, and is child of VoidModule
            for _, value in self.norm_dict[k].items():
                value = value.to(device=device)
                assert (isinstance(value, VoidModule))
        assert len(self.fusion_index_list) == len(self.feature_encoder_dict), \
            "Please check the fusion fusion_index_list ! If you have `n` features, you should have `n` fusion_index !"

    def forward(self, lob_features: dict):
        """ Forward computing of deep-lob net

        :param lob_features: the input feature dict

        :return: predict result, shape=(bs, code, label_len, 1)

        """
        normed_features_dict = {}  # 预处理后的特征
        conved_features_dict = {}  # Conv后的特征
        seqed_features_dict = {}  # Seq后的特征
        mpled_features_dict = {}  # Seq后的特征
        for k in range(len(self.codes)):
            # Split
            normed_features_dict[k] = {}
            # ---- Step 0. To device ---- #
            for key in lob_features.keys():
                if key == "metadata":  # metadata
                    pass
                else:
                    bs_num = lob_features[key].shape[0]
                    seq_num = lob_features[key].shape[2]
                    normed_features_dict[k][key] = [lob_features[key][:, k:k + 1].to(device=self.device, dtype=torch.float32)]

            # ---- Step 2. Read Data and do Normalization ---- #
            for key, norm_module in self.norm_dict[k].items():
                norm_module(normed_features_dict[k])

            # ---- Step 1. Create Fuse Memory ---- #
            x_shape = (bs_num, self.fusion_channel_num, seq_num, 1)  # (bs, channelnum, time_steps, 1)
            conved_features_dict[k] = torch.zeros(x_shape, device=self.device)

            # ---- Step 3. Use Encoder-Dict to do Feature Encoding and Fuse all features --- #
            fusion_index = 0
            for feature_type in self.feature_encoder_dict.keys():
                conved_features_dict[k][:, self.fusion_index_list[fusion_index]] += \
                    self.feature_encoder_dict[feature_type](normed_features_dict[k][feature_type][-1])
                fusion_index += 1

            # ---- Step 4. Seq Encoding ---- #
            # transpose from (bs, c, time_steps, 1) to (bs, time_steps, c, 1)
            conved_features_dict[k] = conved_features_dict[k].permute(0, 2, 1, 3)
            # cut off the last dim to (bs, time_steps, c)
            conved_features_dict[k] = torch.reshape(conved_features_dict[k], (-1, conved_features_dict[k].shape[1], conved_features_dict[k].shape[2]))
            # do the lstm and get the output of each time step
            # the init state DEFAULT be ALL zeros, no matter the type of seq_encoder
            seqed_features_dict[k], _ = self.seq_encoder(conved_features_dict[k])  # shape=(bs, time_steps, hidden_size)
            # get the output of the last label_len steps
            seqed_features_dict[k] = seqed_features_dict[k][:, -self.LS:, :]  # shape=(bs, label_len, lstm_hidden_size)

            # ---- Step 4. Fully Connected ---- #
            # fully connected and get the final output, shape=(bs, label_len, 1)
            mpled_features_dict[k] = self.fc_dict[f"{k}"](seqed_features_dict[k])
            mpled_features_dict[k] = torch.unsqueeze(mpled_features_dict[k], 1)
        # 合并
        y_pred = torch.cat(list(mpled_features_dict.values()), dim=1)
        return {"label": y_pred}
