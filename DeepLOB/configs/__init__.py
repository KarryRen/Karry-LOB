# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2023/12/14 10:25

""" The config version controlling interface.

Now we follow the dynamic configuration principle of config and
    use the SAME config class for rendering and dynamic initialization for different items in ONE `dict_templates`.
    Each config_x_y_z.py file covers a `dict_templates` and config class. The config class might be different.

NOTE:
    - If you want to change the version of the config, you need change the `config_version` in `environ_config` firstly.
        Please check the `README.md` to learn how to change the `config_version`
    - Some configs are not supported now, you can check the `CHANGELOG.md`
        to get the version controlling information.

"""

import os
import yaml
from typing import Dict, Tuple
import importlib
from jinja2 import Environment, FileSystemLoader, DictLoader


def load_yamls(template_name: str, dict_templates: Dict[str, str]) -> Tuple[dict, str]:
    """ Load all yamls config of `template_name`. The yaml might be loaded from 2 location:
        - the `dict_templates` in `config_x_y_z.py`
        - the `./yaml/template_name.yaml`

    :param template_name: the `a` of config_version
    :param dict_templates: the dict of templates in `config_x_y_z.py` file

    return:
        - config_yamls: a dict of config
            {
                key_of_config_1: config_1,
                key_of_config_2: config_2,
                ...,
            }
        - yaml_output: the str of `config_yamls`

    """

    # ---- Step 1. Read and render the yaml to `str` ---- #
    if template_name in dict_templates.keys():  # load from `config_x_y_z.py`
        env = Environment(loader=DictLoader(dict_templates))
        template = env.get_template(template_name)
        yaml_output = template.render()
    else:  # load from `DeepLOB/yaml/template_name.yaml`
        template_path = f"DeepLOB/yaml/{template_name}.yaml"
        assert (os.path.exists(template_path)), f"{template_path} is not existed, please check your config settings !!!"
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template(template_path)
        yaml_output = template.render()

    # ---- Step 2. Load the yaml `str` to `dict` ---- #
    config_yamls = yaml.safe_load(yaml_output)
    if "anchors" in config_yamls.keys():  # pop the `anchors` in config_yamls
        config_yamls.pop("anchors")
    return config_yamls, yaml_output


def load_config(config_version: str) -> Tuple:
    """ Load the config class and config yaml dict to set config.

    :param config_version: the config version, format should be `x.y.z.a.b`
        - a: the template_name
        - b: specific config index

    return:
        - config_cls: the config class
        - config_yamls: the config yaml

    """

    # ---- Import the config based one the `config_version` ---- #
    module_name = f"config_{'_'.join(config_version.split('.')[:3])}"  # only first 3 digit for module name
    module_config = importlib.import_module(name=f".{module_name}", package=__name__)  # import the target config file
    template_params = config_version.split(".")[3:]  # get the [a, b] of the template (x.y.z.a.b)

    # ---- Load the config yamls (might have more than 1 config) basd on `a` ---- #
    template_name = template_params[0]  # get the `a` as template_name
    config_yamls, yaml_output = load_yamls(template_name, module_config.dict_templates)

    # ---- Identify to ONE of the specific config based on `b` ---- #
    if len(template_params) > 1:  # have `b` in `config_version`
        b = int(template_params[1])
        key = list(config_yamls.keys())[b]  # extracted the target key
        config_yamls = {key: config_yamls[key]}  # set the `config_yamls` to the ONLY 1 config

    # ---- Return the config class and yamls ---- #
    config_cls = module_config.Config  # get the config class
    return config_cls, config_yamls
