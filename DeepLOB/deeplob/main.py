# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/21 20:01

""" Train, Valid and Test interface of deeplob.

There are some main operations:
    - create_dirs_log(): Create the directory based on config while setting the logging file.
    - analyze_running_bool(): Analyze the running bool based on the file existing to control repeat running.
    - load_data_dict(): Load data dict to the memory.
    - init_model(): Create the model and init the weight of model to do the pretrain operation.
    - init_optimizer(): Init the optimizer and freeze some params.
    - init_dataloader(): Init the dataloader (Train, Valid and Test are different.)
    - train_valid_model() or test_model(): For epoch and for iter to Train, Valid or Test.

"""

import json
import logging
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import boomdata
from typing import Dict, Tuple, Union, List

from .datasets import get_class_of_dataset, get_class_of_collate
from .datadict.deeplob_datadict import DataDict
from .model import get_instance_of_net
from .model.loss import get_loss_instance
from .model.metrics import r2_score, corr_score
from .utils import fix_random_seed, load_data_dict, analyze_running_bool
from .utils import get_best_state_dict_path, save_feature_as_h5
from .utils import isin_patterns
from .utils import load_patterns_state_dict


def create_dirs_log(config) -> None:
    """ Create the directory based on config while setting the logging file. """

    # ---- Build the save directory ---- #
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # ---- Set logging ---- #
    # get loger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # get handler
    handler = logging.FileHandler(config.LOG_FILE)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    # remove paster handler (from the second config)
    if len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    # add the handler
    logger.addHandler(handler)


def init_model(
        device: torch.device,
        codes: Union[list, str],
        net_config: dict,
        seed_dict: Dict[str, int] = None,
        pretrain: dict = None,
        use_valid_beta: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """ Create the model and init the weight of model.

    :param device: the device to init model
    :param codes: the codes
    :param net_config: the net_config
    :param seed_dict: the seed dict
    :param pretrain: the config of pretrain model, if the path is not None, will load the pretrain weight
    :param use_valid_beta: during test model, whether to use the `beta` estimated by valid

    return:
        - model: the model to use for Train or Test
        - pretrain_model_state_dict: the dict of pretrain model (not empty only the `pretrain` is not None)

    NOTE: There will be 4 main steps during the init model:
        1. Fix the seed of init model, making sure the repeatability of model init.
        2. Build up the net instance based on the interface.
        3. Init the model by the pretrain param if `pretrain` is not None。
        4. Using the valid beta。

    """

    # ---- Fix the seed of init model --- #
    init_model_seed = seed_dict.get("Model")
    logging.info(f"-- Set SEED during BUILDING MODEL as `{init_model_seed}`.")
    if init_model_seed:
        fix_random_seed(seed=init_model_seed)

    # ---- Build up the net instance ---- #
    # Attention: during building the net instance, weight of net is set randomly
    # so the `init_model_seed` is really important, you should set the seed dict carefully.
    model = get_instance_of_net(device=device, class_type=None, codes=codes, **net_config)

    # ---- Init the param of model ---- #
    if pretrain:  # load the state dict of pretrain model, if given the `pretrain_path`
        logging.info(f"||| PreTrain : {pretrain}. |||")
        pretrain_model_state_dict = torch.load(pretrain["path"], map_location=device)  # get the OrderedDict {param_name: param_value}
        if "patterns" in pretrain.keys():
            info = load_patterns_state_dict(model, pretrain_model_state_dict, pretrain["patterns"])
        else:
            info = model.load_state_dict(pretrain_model_state_dict)
        logging.info(f"||| Load param from pretrain model info: {info}. |||")
    else:
        pretrain_model_state_dict = {}

    # ---- Use the valid beta ---- #
    if use_valid_beta:
        # lode the model config
        config_path = pretrain["path"].replace("statedict", "config").replace("pkl", "json")
        with open(config_path, "r", encoding="utf-8") as file:
            model_config = json.load(file)
        # lode the valid beta
        logging.info(f"apply valid_beta {model_config['valid_beta']}")
        if isinstance(model_config["valid_beta"], list):
            if hasattr(model, "fc_dict"):
                for k in range(len(model_config["valid_beta"])):
                    model.fc_dict[f"{k}"].weight.data *= model_config["valid_beta"][k]
            elif hasattr(model, "valid_beta"):
                for k in range(len(model_config["valid_beta"])):
                    model.net_dict[f"{k}"].fc.weight.data *= model_config["valid_beta"][k]
            else:
                raise NotImplementedError("do not know to apply valid beta")
        else:
            model.fc.weight.data *= model_config["valid_beta"]

    return model, pretrain_model_state_dict


def init_optimizer(
        config,
        model: torch.nn.Module,
        pretrain_model_state_dict: Dict
) -> torch.optim.Optimizer:
    """ Create the optimizer while do freezing.

    :param config: the config file
    :param model: the model to use for Train or Test
    :param pretrain_model_state_dict: the dict of pretrain model (not empty only the `pretrain_path` is not None)

    """

    # ---- Freeze the params ---- #
    freeze_patterns = config.OPTIM_PARAMS.get("freeze")  # get the freeze config
    assert isinstance(freeze_patterns, list), "The `freeze` in config must be a list !"
    if freeze_patterns:  # You can't freeze all params !
        for param_name, param in model.named_parameters():
            if isin_patterns(param_name, freeze_patterns):
                if param_name in pretrain_model_state_dict.keys():
                    # if the name of params is same but shape is different, you should erase it in config
                    assert param.shape == pretrain_model_state_dict[param_name].shape, f"{param_name} shape error ! Can't Freeze !"
                    # freeze the param
                    param.requires_grad = False
                    logging.info(f"-- Freeze param `{param_name}` in model ! --")

    # ---- Freeze the mean&var in BN ---- #
    for pretrain_model_param_key in pretrain_model_state_dict.keys():
        if "norm_dict.BN_" in pretrain_model_param_key:
            feature_key = pretrain_model_param_key.split(".")[1]
            model.norm_dict[feature_key].model.bn.momentum = 0
            logging.info(f"-- Freeze bn running mean&var of `{pretrain_model_param_key}` in model ! --")

    # ---- Check whether all params are freezing ---- #
    no_freeze_param_list = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(no_freeze_param_list) > 0, "You freeze all params in model ! Not Allowed !"

    # ---- Create the optimizer --- #
    if config.OPTIM_PARAMS["cls_kwargs"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **config.OPTIM_PARAMS["init_kwargs"])
    elif config.OPTIM_PARAMS["cls_kwargs"]["name"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), **config.OPTIM_PARAMS["init_kwargs"])
    else:
        raise ValueError(f"The Type of Optimize {config.OPTIM_PARAMS['cls_kwargs']['name']} is not allowed now.")
    return optimizer


def init_train_valid_dataloader(
        device: torch.device,
        config,
        seed_dict: Dict[str, int],
        data_dict: DataDict
) -> dict:
    """ Init the train and valid dataloader by data_dict.

    :param device: the data device after `collate_fn()`
    :param config: the config file
    :param seed_dict: the seed dict
    :param data_dict: the grouped data dict after `load_data_dict()`

    return:
        - the dict of train_valid_dataloader

    NOTE:
        - The dataloader of training has some tricks, please be careful.
        - seed_dict effects when shuffle=True.
    """

    # ---- Fix the seed of init train and valid data loader --- #
    init_tv_dataloader_seed = seed_dict.get("DataLoader")
    logging.info(f"-- Set SEED during CONSTRUCTING TRAIN & VALID DATALOADER as `{init_tv_dataloader_seed}`.")
    if init_tv_dataloader_seed:
        fix_random_seed(seed=init_tv_dataloader_seed)

    # ---- Logging ---- #
    logging.info(f"***************** BEGIN MAKE DATASET ! *****************")
    logging.info(f"||| needed_features = {config.NEEDED_FEATURES} |||")
    logging.info(f"||| feature_len = {config.FEATURE_LEN}, label_len = {config.LABEL_LEN} |||")
    logging.info(f"||| paf_shift_k = {config.PAF_SHIFT_K} |||")

    # ---- Build up the Train dataset and loader (Might have two types) ---- #
    train_dataset_class = get_class_of_dataset(class_type=config.TRAIN_DATASET_TYPE, codes=config.FUTURES)
    train_collate_fn = get_class_of_collate(class_type=config.TRAIN_DATASET_TYPE)(device=device)
    logging.info(f"**** TRAIN FROM {config.TRAIN_DATES[0]} TO {config.TRAIN_DATES[-1]} ! ****")
    logging.info(f"||| train dataset type = {config.TRAIN_DATASET_TYPE} |||")
    if config.TRAIN_DATASET_TYPE == "DeepLOBDataset_DailySample":
        train_dataset = train_dataset_class(
            data_root_dict=data_dict,
            dates=config.TRAIN_DATES,
            feature_len=config.FEATURE_LEN,
            label_len=config.LABEL_LEN,
            start_tick=config.START_TICK,
            end_tick=config.END_TICK,
            tick_num_in_one_sample=config.TICK_NUM_IN_ONE_SAMPLE,
            paf_shift_k=config.PAF_SHIFT_K,
            use_sample_std_weight=config.DICT.get("use_sample_std_weight", False)
        )
        batch_size = config.TRAIN_BATCH_SIZE
        logging.info(f"||| sec num in one sample = {config.TICK_NUM_IN_ONE_SAMPLE} |||")
        logging.info(f"||| train batch size = {config.TRAIN_BATCH_SIZE} |||")
    else:
        train_dataset = train_dataset_class(
            data_root_dict=data_dict,
            dates=config.TRAIN_DATES,
            feature_len=config.FEATURE_LEN,
            label_len=config.LABEL_LEN,
            start_tick=config.START_TICK,
            end_tick=config.END_TICK,
            paf_shift_k=config.PAF_SHIFT_K,
            use_sample_std_weight=config.DICT.get("use_sample_std_weight", False)
        )
        batch_size = config.BATCH_SIZE
        logging.info(f"||| batch size = {config.BATCH_SIZE} |||")
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)

    # ---- Build up the Valid dataset and loader (Might have two types) ---- #
    logging.info(f"**** VALID FROM {config.VALID_DATES[0]} TO {config.VALID_DATES[-1]} ! ****")
    valid_dataset_class = get_class_of_dataset(codes=config.FUTURES)
    valid_collate_fn = get_class_of_collate()(device=device)
    valid_dataset = valid_dataset_class(
        data_root_dict=data_dict,
        dates=config.VALID_DATES,
        feature_len=config.FEATURE_LEN,
        start_tick=config.START_TICK,
        end_tick=config.END_TICK,
        paf_shift_k=config.PAF_SHIFT_K,
        use_sample_std_weight=config.DICT.get("use_sample_std_weight", False)
    )
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=valid_collate_fn)

    # ---- Build up over ---- #
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")
    return {"train": train_loader, "valid": valid_loader}


def train_valid_model(
        config,
        seed_dict: Dict[str, int],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader_dict: dict
):
    """ Train and valid the model.

    :param config: the config file
    :param seed_dict: the seed dict
    :param model: the model to train & valid
    :param optimizer: the optimizer
    :param data_loader_dict: the dict of data loader, format should be {"train": train_loader, "valid": valid_loader}

    """

    logging.info(f"***************** RUN TRAIN&VALID DEEP-LOB ! *****************")

    # ---- Get the data loader ---- #
    train_loader, valid_loader = data_loader_dict["train"], data_loader_dict["valid"]

    # ---- Make the preparation for train&valid ---- #
    # the loss function
    criterion = get_loss_instance(config.DICT.get("loss", {"MTL": {}}))
    # init the metric dict (will note `epochs+1` number metrics)
    df_metric = {}
    train_needed_metrics = ["loss", "global_r2", "global_corr"]
    valid_needed_metrics = ["loss", "global_r2", "global_corr", "daily_mid_r2", "beta", "rescaled_global_r2", "rescaled_daily_mean_r2",
                            "rescaled_global_corr"]
    for needed_metric in train_needed_metrics:
        df_metric[f"train_{needed_metric}"] = np.zeros(config.EPOCHS + 1)
    for needed_metric in valid_needed_metrics:
        df_metric[f"valid_{needed_metric}"] = np.zeros(config.EPOCHS + 1)

    # ---- Valid the initial model ---- #
    valid_loss_one_epoch, valid_preds_one_epoch, valid_labels_one_epoch, valid_weights_one_epoch = valid_model_one_epoch(
        config=config, model=model, valid_loader=valid_loader, criterion=criterion
    )
    summary_model_result_one_epoch(
        df_metric=df_metric, epoch=0, config=config,
        loss_one_epoch=valid_loss_one_epoch, preds_one_epoch=valid_preds_one_epoch,
        labels_one_epoch=valid_labels_one_epoch, weights_one_epoch=valid_weights_one_epoch,
        summary_type="valid", needed_metrics=valid_needed_metrics
    )
    save_model_one_epoch(model=model, epoch=0, root_path=config.MODEL_SAVE_PATH)
    logging.info(f"Init: {['%s:%.4f ' % (key, value[0]) for key, value in df_metric.items()]}")

    # ---- Fix the seed of train&valid model --- #
    train_valid_seed = seed_dict.get("Train")
    logging.info(f"-- Set SEED during TRAIN & VALID MODEL as `{train_valid_seed}`.")
    if train_valid_seed:
        fix_random_seed(seed=train_valid_seed)

    # ---- Start Train & Valid epoch by epoch ---- #
    for epoch in tqdm(range(1, config.EPOCHS + 1)):
        # start timer for one epoch
        t_start = datetime.now()
        # train model for one epoch
        train_loss_one_epoch, train_preds_one_epoch, train_labels_one_epoch, train_weights_one_epoch = train_model_one_epoch(
            config=config, model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion
        )
        summary_model_result_one_epoch(
            df_metric=df_metric, epoch=epoch, config=config,
            loss_one_epoch=train_loss_one_epoch, preds_one_epoch=train_preds_one_epoch,
            labels_one_epoch=train_labels_one_epoch, weights_one_epoch=train_weights_one_epoch,
            summary_type="train", needed_metrics=train_needed_metrics
        )
        # valid model for one epoch
        valid_loss_one_epoch, valid_preds_one_epoch, valid_labels_one_epoch, valid_weights_one_epoch = valid_model_one_epoch(
            config=config, model=model, valid_loader=valid_loader, criterion=criterion
        )
        summary_model_result_one_epoch(
            df_metric=df_metric, epoch=epoch, config=config,
            loss_one_epoch=valid_loss_one_epoch, preds_one_epoch=valid_preds_one_epoch,
            labels_one_epoch=valid_labels_one_epoch, weights_one_epoch=valid_weights_one_epoch,
            summary_type="valid", needed_metrics=valid_needed_metrics
        )
        # save the model
        save_model_one_epoch(model=model, epoch=epoch, root_path=config.MODEL_SAVE_PATH)
        # end the timer for one epoch
        dt = datetime.now() - t_start
        # log the information
        logging.info(f"Epoch {epoch}/{config.EPOCHS}, Duration: {dt}, {['%s:%.4f ' % (key, value[epoch]) for key, value in df_metric.items()]}")

    # ---- Summary the train & valid result ---- #
    # save the metric dict to csv
    pd.DataFrame(df_metric).to_csv(config.MODEL_SAVE_PATH + "model_pytorch_metric.csv")
    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.plot(df_metric["train_loss"], label="train loss", color="g")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(df_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt_r2 = [key for key in df_metric.keys() if "_r2" in key]
    color = config.DICT.get("color", ["darkorange", "forestgreen", "slategrey", "royalblue", "crimson", "dodgerblue", "yellow", "tomato"])
    for k, key in enumerate(plt_r2):
        plt.plot(df_metric[key], label=key, color=color[k], linestyle="--")
    plt.legend()
    plt.savefig(config.SAVE_PATH + "training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************")


def train_model_one_epoch(
        config,
        model: torch.nn.Module,
        train_loader: data.dataloader,
        optimizer: torch.optim.Optimizer,
        criterion,
) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Train the model for one epoch.

    :param config: the config file
    :param model: the model to be trained
    :param train_loader: the dataloader for training model
    :param optimizer: the optimizer
    :param criterion: the loss instance

    return:
        - train_loss_one_epoch: the summary loss, length=(num_of_iter)
        - train_preds_one_epoch: the summary prediction
        - train_labels_one_epoch: the summary label
        - train_weights_one_epoch: the summary weight

    """

    # ---- Step 1. Do some preparation for train model ---- #
    device = model.device  # get the computing device
    train_loss_one_epoch = []  # set the empty loss list
    if config.TRAIN_DATASET_TYPE == "DeepLOBDataset_DailySample":
        real_sample_num = len(train_loader.dataset) * config.TICK_NUM_IN_ONE_SAMPLE * len(config.TRAIN_DATES)
    else:
        real_sample_num = len(train_loader.dataset)
    if isinstance(config.FUTURES, list):
        train_preds_one_epoch = torch.zeros(real_sample_num, len(config.FUTURES)).to(device=device)
    else:
        train_preds_one_epoch = torch.zeros(real_sample_num).to(device=device)
    train_labels_one_epoch = torch.zeros_like(train_preds_one_epoch)
    train_weights_one_epoch = torch.zeros_like(train_preds_one_epoch)

    # ---- Step 2. Train the model iter by iter ---- #
    last_step = 0
    model.train()
    for batch_data, batch_label in tqdm(train_loader):
        # read the data
        lob_features = batch_data
        lob_labels, lob_weights = batch_label["label"], batch_label["weight"]
        # zero_grad, forward, compute loss, backward and optimize
        optimizer.zero_grad()
        outputs = model(lob_features, batch_label)["label"]
        loss = criterion(outputs, lob_labels, lob_weights)
        loss.backward()
        optimizer.step()
        # note the loss of training in one iter
        train_loss_one_epoch.append(loss.item())
        # doc the result in one iter, no matter what label_len is, just get the last one
        now_step = last_step + outputs.shape[0]
        train_preds_one_epoch[last_step:now_step] = outputs[..., -1, 0].detach()
        train_labels_one_epoch[last_step:now_step] = lob_labels[..., -1, 0].detach()
        train_weights_one_epoch[last_step:now_step] = lob_weights[..., -1, 0].detach()
        last_step = now_step
    return train_loss_one_epoch, train_preds_one_epoch, train_labels_one_epoch, train_weights_one_epoch


def valid_model_one_epoch(
        config,
        model: torch.nn.Module,
        valid_loader: data.dataloader,
        criterion,
) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Valid the model for one epoch.

    :param config: the config file
    :param model: the model to be trained
    :param valid_loader: the dataloader for valid model
    :param criterion: the loss instance

    return:
        - valid_loss_one_epoch: the summary loss, length=(num_of_iter)
        - valid_preds_one_epoch: the summary prediction
        - valid_labels_one_epoch: the summary label
        - valid_weights_one_epoch: the summary weight

    """

    # ---- Step 1. Do some preparation for valid model ---- #
    device = model.device  # get the computing device
    valid_loss_one_epoch = []  # set the empty loss list
    if isinstance(config.FUTURES, list):
        valid_preds_one_epoch = torch.zeros(len(valid_loader.dataset), len(config.FUTURES)).to(device=device)
    else:
        valid_preds_one_epoch = torch.zeros(len(valid_loader.dataset)).to(device=device)
    valid_labels_one_epoch = torch.zeros_like(valid_preds_one_epoch)
    valid_weights_one_epoch = torch.zeros_like(valid_preds_one_epoch)

    # ---- Step 2. Valid the model iter by iter ---- #
    last_step = 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_label in tqdm(valid_loader):
            # read the data
            lob_features = batch_data
            lob_labels, lob_weights = batch_label["label"], batch_label["weight"]
            # forward to compute outputs
            outputs = model(lob_features)["label"]
            # note the loss of valid in one iter
            loss = criterion(outputs, lob_labels, lob_weights)
            valid_loss_one_epoch.append(loss.item())
            # doc the result in one iter, no matter what label_len is, just get the last one
            now_step = last_step + outputs.shape[0]
            valid_preds_one_epoch[last_step:now_step] = outputs[..., -1, 0].detach()
            valid_labels_one_epoch[last_step:now_step] = lob_labels[..., -1, 0].detach()
            valid_weights_one_epoch[last_step:now_step] = lob_weights[..., -1, 0].detach()
            last_step = now_step
    return valid_loss_one_epoch, valid_preds_one_epoch, valid_labels_one_epoch, valid_weights_one_epoch


def summary_model_result_one_epoch(
        df_metric: Dict[str, np.ndarray],
        epoch: int,
        config,
        loss_one_epoch: list,
        preds_one_epoch: torch.Tensor,
        labels_one_epoch: torch.Tensor,
        weights_one_epoch: torch.Tensor,
        summary_type: str,
        needed_metrics: List[str]
) -> None:
    """ Summary model result of one epoch.

    :param df_metric: the summary dataframe of metrics
    :param epoch: the epoch idx
    :param config: the config file
    :param loss_one_epoch: the loss list
    :param preds_one_epoch: the pred tensor
    :param labels_one_epoch: the label tensor
    :param weights_one_epoch: the weight tensor
    :param summary_type: the summary type
    :param needed_metrics: the needed metrics list

    return: None

    """

    # ---- Summary the beta for valid ---- #
    if summary_type == "valid":
        # compute the valid beta
        if len(preds_one_epoch.shape) == 1:
            x_dot_x = torch.sum(preds_one_epoch.view(-1) * preds_one_epoch.view(-1) * weights_one_epoch.view(-1))
            x_dot_y = torch.sum(preds_one_epoch.view(-1) * labels_one_epoch.view(-1) * weights_one_epoch.view(-1))
            beta = (x_dot_y / x_dot_x).cpu().numpy()
        else:
            x_dot_x = torch.sum(preds_one_epoch * preds_one_epoch * weights_one_epoch, dim=0)
            x_dot_y = torch.sum(preds_one_epoch * labels_one_epoch * weights_one_epoch, dim=0)
            beta = (x_dot_y / x_dot_x).cpu().numpy()
        df_metric[f"valid_beta"][epoch] = 0.0 if len(beta.shape) > 0 else float(beta)
        # save model config (only valid_beta makes sense)
        if len(beta.shape) > 0:  # multi code
            model_config = {"valid_beta": [float(x) for x in beta]}
        else:
            model_config = {"valid_beta": float(beta)}
        with open(config.MODEL_SAVE_PATH + f"model_config_epoch_{epoch}.json", "w") as file:
            json.dump(model_config, file)
    else:
        beta = 1.0

    # ---- Summary the metric ---- #
    for needed_metric in needed_metrics:
        if needed_metric == "loss":
            df_metric[f"{summary_type}_loss"][epoch] = np.mean(loss_one_epoch)
        elif needed_metric == "global_r2":
            df_metric[f"{summary_type}_global_r2"][epoch] = r2_score(
                y_true=labels_one_epoch.cpu().numpy(), y_pred=preds_one_epoch.cpu().numpy(), weight=weights_one_epoch.cpu().numpy()
            )
        elif needed_metric == "global_corr":
            df_metric[f"{summary_type}_global_corr"][epoch] = corr_score(
                y_true=labels_one_epoch.cpu().numpy(), y_pred=preds_one_epoch.cpu().numpy(), weight=weights_one_epoch.cpu().numpy()
            )
        elif needed_metric == "daily_mid_r2":
            df_metric[f"{summary_type}_daily_mid_r2"][epoch] = r2_score(
                y_true=labels_one_epoch.cpu().numpy().T, y_pred=preds_one_epoch.cpu().numpy().T, weight=weights_one_epoch.cpu().numpy().T,
                mode="daily_mid", tick_num=config.TICK_NUM,
            )
        elif needed_metric == "rescaled_global_r2":
            df_metric[f"{summary_type}_rescaled_global_r2"][epoch] = r2_score(
                y_true=labels_one_epoch.cpu().numpy(), y_pred=preds_one_epoch.cpu().numpy() * beta, weight=weights_one_epoch.cpu().numpy()
            )
        elif needed_metric == "rescaled_global_corr":
            df_metric[f"{summary_type}_rescaled_global_corr"][epoch] = r2_score(
                y_true=labels_one_epoch.cpu().numpy(), y_pred=preds_one_epoch.cpu().numpy() * beta, weight=weights_one_epoch.cpu().numpy()
            )
        elif needed_metric == "rescaled_daily_mean_r2":
            df_metric["valid_rescaled_daily_mean_r2"][epoch] = r2_score(
                y_true=labels_one_epoch.cpu().numpy().T, y_pred=(preds_one_epoch.cpu().numpy() * beta).T, weight=weights_one_epoch.cpu().numpy().T,
                mode="daily_mean", tick_num=config.TICK_NUM,
            )
        elif needed_metric == "beta":  # have computed in the top, just pass
            pass
        else:
            raise TypeError(needed_metric)


def save_model_one_epoch(
        model: torch.nn.Module,
        epoch: int,
        root_path: str
) -> None:
    """ Save the model.

    :param model: the model to be saved
    :param epoch: the idx of epoch
    :param root_path: the root path of saving model

    """

    torch.save(model.state_dict(), root_path + f"model_statedict_epoch_{epoch}.pkl")


def test_model(model, data_root_dict, config):
    """
    test. no seed.
    """

    logging.info(f"***************** BEGIN MAKE DATASET ! *****************")
    device = model.device  # get the device

    # ---- Load test data, then make dataset following by dataloader ---- #
    logging.info(f"**** TEST FROM {config.TEST_DATES[0]} TO {config.TEST_DATES[-1]} ! ****")
    test_dataset_class = get_class_of_dataset(codes=config.FUTURES)
    test_collate_fn = get_class_of_collate()(device=device)
    test_dataset = test_dataset_class(
        data_root_dict=data_root_dict,
        dates=config.TEST_DATES,
        feature_len=config.FEATURE_LEN,
        label_len=config.LABEL_LEN,
        start_tick=config.START_TICK,
        end_tick=config.END_TICK,
        paf_shift_k=config.PAF_SHIFT_K
    )
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=test_collate_fn)
    logging.info("***************** LOAD DATA OVER ! *****************")
    logging.info(f"Test dataset: length = {len(test_dataset)}")

    # ---- Init Feature for saving predict ---- #
    feature_dict = {
        "codes": config.FUTURES if isinstance(config.FUTURES, list) else [config.FUTURES],
        "dates": config.TEST_DATES,
        "timestamps": boomdata.util.generate_ts("[('09:30:00', '11:30:00'], ('13:00:00', '15:00:00']]", "500ms"),
        "features": ["Weight", "Label", "Pred"],
        "ftype": 'numpy.array(code, date, timestamp, feature)',
        "data": np.nan
    }
    fea_label_pred = boomdata.Feature(**feature_dict)

    # ---- Test model ---- #
    if isinstance(config.FUTURES, list):
        lob_labels_array = torch.zeros(len(test_dataset), len(config.FUTURES)).to(device=device)
    else:
        lob_labels_array = torch.zeros(len(test_dataset)).to(device=device)
    predictions_array = torch.zeros_like(lob_labels_array)
    weight_array = torch.zeros_like(lob_labels_array)
    last_step = 0
    model.eval()  # start test
    with torch.no_grad():
        for batch_data, batch_label in tqdm(test_loader):
            # read the data
            lob_features = batch_data
            lob_labels, lob_weights = batch_label["label"], batch_label["weight"]
            # forward
            outputs = model(lob_features)["label"]
            # doc the result
            now_step = last_step + outputs.shape[0]
            lob_labels_array[last_step:now_step] = lob_labels[..., -1, 0].detach()
            predictions_array[last_step:now_step] = outputs[..., -1, 0].detach()
            weight_array[last_step:now_step] = lob_weights[..., -1, 0].detach()
            last_step = now_step

    # ---- Four Metrics ---- #
    lob_labels_array_global = lob_labels_array.cpu().numpy()
    predictions_array_global = predictions_array.cpu().numpy()
    weight_array_global = weight_array.cpu().numpy()
    lob_labels_array_daily = lob_labels_array_global.reshape(len(config.TEST_DATES), -1)
    predictions_array_daily = predictions_array_global.reshape(len(config.TEST_DATES), -1)
    weight_array_daily = weight_array_global.reshape(len(config.TEST_DATES), -1)
    # logging
    logging.info(f"******** R2_Global : "
                 f"{r2_score(y_true=lob_labels_array_global, y_pred=predictions_array_global, weight=weight_array_global, mode='global')} "
                 f"**********")
    logging.info(f"******** R2_Daily_Mid : "
                 f"{r2_score(y_true=lob_labels_array_daily, y_pred=predictions_array_daily, weight=weight_array_daily, mode='daily_mid', tick_num=config.TICK_NUM)} "
                 f"**********")
    logging.info(f"******** Corr_Global : "
                 f"{corr_score(y_true=lob_labels_array_global, y_pred=predictions_array_global, weight=weight_array_global, mode='global')} "
                 f"**********")
    logging.info(f"******** Corr_Daily_Mid : "
                 f"{corr_score(y_true=lob_labels_array_daily, y_pred=predictions_array_daily, weight=weight_array_daily, mode='daily_mid', tick_num=config.TICK_NUM)} "
                 f"**********")
    logging.info("***************** TEST OVER ! *****************")
    logging.info("")
    # cat
    if len(weight_array_global.shape) == 1:
        concat_array = np.concatenate((weight_array_global[:, None], lob_labels_array_global[:, None], predictions_array_global[:, None]), axis=1)
    else:
        concat_array = np.concatenate((np.expand_dims(weight_array_global.T, -1), np.expand_dims(lob_labels_array_global.T, -1),
                                       np.expand_dims(predictions_array_global.T, -1)), axis=-1)
    fea_label_pred.data[:] = concat_array.reshape(*fea_label_pred.shape)

    # set zero
    fea_label_pred.data[:, :, :config.START_TICK, 2] = 0.0
    fea_label_pred.data[:, :, config.END_TICK:, 2] = 0.0
    # save
    save_feature_as_h5(fea_label_pred, config.SAVE_PATH + "label_pred.h5")


def run(config) -> None:
    """ The main interface of deeplob. If you want to use our deeplob you can just import this interface. """

    # ---- Create needed directories and set logger ---- #
    create_dirs_log(config)

    # ---- Analyze running bool ---- #
    running_bool = analyze_running_bool(config, config.REDO, ["training_steps.png", "label_pred.h5"])
    print(f"RUNNING BOOL: {running_bool}")

    # ---- Load the `device` and `data_dict` ---- #
    if running_bool["Train"] or running_bool["Test"]:
        # - load the device
        device = torch.device(config.DEVICE)
        logging.info(f"***************** The deeplob will compute in device: `{device}`   *****************")
        # - load the data_dict
        needed_data = {}  # define the empty needed_data
        needed_data.update(config.NEEDED_FEATURES)  # set the features config
        needed_data.update(config.NEEDED_LABELS)  # set the labels config
        data_dict = load_data_dict(
            dates=config.TRAIN_DATES + config.VALID_DATES + config.TEST_DATES,
            codes=config.FUTURES, needed_data=needed_data, data_root_path_dict=config.DATA_ROOT_DICT
        )

    # ---- Train & Valid model ---- #
    if running_bool["Train"]:
        model, pretrain_model_state_dict = init_model(
            device=device, codes=config.FUTURES, net_config=config.NET, seed_dict=config.SEED_DICT, pretrain=config.NET.get("pretrain")
        )
        optimizer = init_optimizer(config=config, model=model, pretrain_model_state_dict=pretrain_model_state_dict)
        train_valid_data_loader_dict = init_train_valid_dataloader(device=model.device, config=config, seed_dict=config.SEED_DICT,
                                                                   data_dict=data_dict)
        train_valid_model(
            config=config, seed_dict=config.SEED_DICT, model=model, optimizer=optimizer, data_loader_dict=train_valid_data_loader_dict
        )

    # ---- Step 4. Test model ---- #
    if running_bool["Test"]:
        best_path = get_best_state_dict_path(config.MODEL_SAVE_PATH, config.MAIN_METRIC)
        logging.info(f"***************** LOAD Best Model {best_path} *****************")
        model, _ = init_model(
            device=device, codes=config.FUTURES, net_config=config.NET, seed_dict={}, pretrain={"path": best_path},
            use_valid_beta=config.DICT.get("use_valid_beta", False)
        )
        test_model(model=model, data_root_dict=data_dict, config=config)
