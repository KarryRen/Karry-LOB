# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/11/10 12:54

""" Train and test deeplob based of the `config_version` and backup the package code. """

import os
import json

from DeepLOB.task_util import environ_config  # during the import, all in `task_util.py` have been run.
from DeepLOB.configs import load_config

# from deeplob.main import run

if __name__ == "__main__":
    # ---- Get the config_version from os environment ---- #
    config_version = environ_config["config_version"]

    # ---- Load the config based on the config_version ---- #
    config_cls, config_yamls = load_config(config_version)

    # ---- Step 4. Run the train&test deeplob ---- #
    for key, config_yaml in config_yamls.items():
        # the config is real-time rendering because there might be dependencies
        # between the items in config_yaml, such as pretrain_path and so on.
        config = config_cls(config_yaml=config_yaml, job_name=f"{'.'.join(config_version.split('.')[:4])}.{key}")
        # log the information
        print("***************** HERE IS CONFIG INFO ! *****************")
        print(f"Run the deeplob of config {key}")
        print("-- config_yaml: ")
        print(json.dumps(config_yaml, indent=4))
        print("-- config.__dict__: ")
        print(json.dumps(config.__dict__, indent=4))
        print("***************** CONFIG INFO END ! *****************")
        run(config)
