# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/03/15 10:25

""" The main function for construction.

本函数用以生成基础特征.
基础特征通常直接采用基础行情数据, 或者其他基础另类数据计算得到.
它的数据源来源是自由的, 通常的来源有: .csv, 另类数据等.

请不要直接调用 python 运行本文件, 详情查看 README.md

"""

import os
import datetime
import numpy as np
from multiprocessing import Pool
from typing import List

from TensorEngineering.tensor_engineering.util import gen_date_list, get_code_list
from Ops import CodeType
from TensorEngineering.tensor_engineering.io import TensorEngineeringRootDir, get_file_path

# The following 2 import is really important !
from TensorEngineering.tensor_engineering.tensor_construction_algo import ConstructionAlgoBase
import TensorEngineering.tensor_engineering.tensor_construction_algo as Algo

# ---- Collect all algorithm classes (really direct way) ---- #
algo_base_class = "ConstructionAlgoBase"
global_algo_class_list = []
for x in dir(globals()["Algo"]):
    value = getattr(globals()["Algo"], x)
    try:
        is_algo_class = issubclass(value, globals().get(algo_base_class))
    except Exception:
        is_algo_class = False
    if is_algo_class and x != algo_base_class:
        global_algo_class_list.append(value)
        print(f"Going to construct the class: `{value}`")


def tensor_construction_by_date(code_type: CodeType, date_list: List[str], algo_class, mode: str) -> None:
    """ Construct the factor in the data_list of the code_type by algo_class in mode.

    :param code_type: code type
    :param date_list: list of dates
    :param algo_class: algo class
    :param mode: specific mode
        - `a` is for adding mode, there will be a detection for existed factors

    """

    # ---- Get the code & date list ---- #
    code_list = get_code_list(code_type)
    print(f"-- Going to operate `{len(code_list)}` codes & `{len(date_list)}` dates")

    # ---- Build up the function & collect the algorithm detail ---- #
    algo_func = algo_class(code_type=code_type)
    out_ftype = algo_func.out_coords.keys()
    version = algo_func.__version__
    file_name = algo_func.__class__.__name__

    # ---- Make the directories ---- #
    for code in code_list:
        for date in date_list:
            path = f"{TensorEngineeringRootDir}/{code}/{date}"
            os.makedirs(path, exist_ok=True)

    # ---- Judge whether feature is existed or not in mode `a` ---- #
    if mode == "a":
        feature_existed_list = []
        for date in date_list:
            for code in code_list:
                file_path = get_file_path(code, date, file_name)
                bool_existed = os.path.exists(file_path)  # path is existed or not
                if bool_existed:
                    create_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                    if create_time < datetime.datetime(*version):  # version is elder or not
                        bool_existed = False
                feature_existed_list.append(bool_existed)
        if np.all(np.array(feature_existed_list)):
            print(f"!! Feature `{file_name}` is existed. Skip !!")
            return

    # ---- Run the function ---- #
    n_process = int(os.environ.get("n_process", 32))  # set the n_process
    if n_process <= 1:  # `n_process <= 1` for debug, not save the feature
        res_fea_list = []
        if ["T"] == list(out_ftype)[:1]:  # out_ftype == ["T", "..."]
            for d, date in enumerate(date_list):
                for c, code in enumerate(code_list):
                    print(f"- running ({d}, {c})|({len(date_list)}, {len(code_list)}) -- {algo_func}, {code}, {date}")
                    res_fea_list.append(algo_func(code, date, save=False))
        else:
            raise ValueError(out_ftype)
        print(res_fea_list[0])
    else:
        code_date = [(code, date) for code in code_list for date in date_list]
        if ["T"] == list(out_ftype)[:1]:
            with Pool(n_process) as p:
                p.starmap(algo_func, code_date)
        else:
            raise ValueError(out_ftype)


if __name__ == "__main__":
    """ Workflow for construction. """

    # ---- Generate date list ---- #
    start_date, end_date = os.environ.get("start_date"), os.environ.get("end_date")
    date_list = gen_date_list(start_date, end_date)

    # ---- Get the code type ---- #
    code_type = CodeType[os.environ.get("code_type")]

    # ---- Get the mode ---- #
    mode = os.environ.get("mode")

    # ---- Log the info and operate one by one ---- #
    print(f"---- Operating `{len(date_list)}` trading dates, in `[{date_list[-1]}, {date_list[0]}]`")
    for algo_class in global_algo_class_list:
        print(f"--- Using algorithm `{algo_class}`.")
        tensor_construction_by_date(code_type, date_list, algo_class, mode)
