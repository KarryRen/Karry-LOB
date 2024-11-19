# -*- coding: utf-8 -*-
# @Author : Karry Ren
# @Time   : 2024/03/15 10:25

""" The main function for construction.

本函数用以生成基础特征.
基础特征通常直接采用基础行情数据, 或者其他基础另类数据计算得到. 它的数据源来源是自由的, 通常的来源有: .csv, 另类数据等.

"""

import os
import datetime
import numpy as np
from multiprocessing import Pool

from TensorEngineering.tensor_engineering.util import gen_date_list, get_date_by_taskid, get_code_list
from Ops import CodeType
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


def date_to_fea(code_type, date, algo_class, mode):
    """
    获取特定月份的数据，按特征分别保存为h5文件
    """
    func = algo_class(code_type=code_type)
    code_list = get_code_list(code_type, date)
    code_num = len(code_list)
    print('code num:', code_num)
    date_list = [date]
    date_num = len(date_list)
    print('date num:', date_num)
    print('for loop start')
    if os.environ.get('max_code_num'):
        code_slice = int(os.environ.get('max_code_num'))
        code_list = code_list[:code_slice]
    out_ftype = func.out_coords.keys()
    out_features = func.out_coords['F']
    out_timestamps = func.out_coords['T']
    version = func.__version__
    file_name = func.__class__.__name__
    # 创建文件夹
    from TensorEngineering.tensor_engineering.io import TensorEngineeringRootDir, get_file_path
    for date in date_list:
        for code in code_list:
            path = os.path.join(TensorEngineeringRootDir, f"{code}", date, )
            os.makedirs(path, exist_ok=True)
    # 判断特征是否已经存在
    if mode == "a":
        array_exists = []
        for date in date_list:
            for code in code_list:
                file_path = get_file_path(code, date, file_name)
                # 先验证文件是否存在
                bool_exists = os.path.exists(file_path)
                if bool_exists:
                    # 验证生成时间是否满足版本
                    create_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                    if create_time < datetime.datetime(*version):
                        bool_exists = False
                array_exists.append(bool_exists)
        if np.all(np.array(array_exists)):
            print('data exists. skip.', file_name)
            return
    # 此处需要改成并行
    n_process = int(os.environ.get('n_process', 32))
    # n_process = 1
    if n_process <= 1:
        # debug用
        print(func)
        res_list = []
        if ['T'] == list(out_ftype)[:1]:
            for d, date in enumerate(date_list):
                for c, code in enumerate(code_list):
                    print('run (%d,%d)|(%d,%d)' % (d, c, len(date_list), len(code_list)), func, code, date)
                    res_fea = func(code, date)
                    res_list.append(res_fea)
        elif ['C', 'T'] == list(out_ftype)[:2]:
            for d, date in enumerate(date_list):
                print('run (%d)|(%d)' % (d, len(date_list)), func, date)
                res_fea = func(code_list, date)
                res_list.append(res_fea)
        elif ['D', 'T'] == list(out_ftype)[:2]:
            for c, code in enumerate(code_list):
                res_fea = func(code, date_list)
                res_list.append(res_fea)
                print('run (%d)|(%d)' % (c, len(code_list)), func, code, res_fea.shape)
        elif ['C', 'D', 'T'] == list(out_ftype)[:3]:
            res_fea = func(code_list, date_list)
            res_list.append(res_fea)
        else:
            raise ValueError(out_ftype)
        print(res_fea)
        print('debug mode. not save feature')
        # sys.exit()
    else:
        code_date = [(code, date) for code in code_list for date in date_list]
        code_list_date = [(code_list, date) for date in date_list]
        code_date_list = [(code, date_list) for code in code_list]
        if ['C', 'D', 'T'] == list(out_ftype)[:3]:
            res_list = [func(code_list, date_list)]
        elif ['C', 'T'] == list(out_ftype)[:2]:
            with Pool(n_process) as p:
                res0 = p.starmap(func, code_list_date)
        elif ['D', 'T'] == list(out_ftype)[:2]:
            with Pool(n_process) as p:
                res0 = p.starmap(func, code_date_list)
        elif ['T'] == list(out_ftype)[:1]:
            with Pool(n_process) as p:
                res0 = p.starmap(func, code_date)
        else:
            raise ValueError(func.out_ftype)
    return


if __name__ == "__main__":
    """ Workflow for construction. """

    # ---- Generate date list ---- #
    start_date, end_date = os.environ.get("start_date"), os.environ.get("end_date")
    date_list = gen_date_list(start_date, end_date)
    print(date_list)

    # 根据以下两个变量，生成一个date_list
    code_type = CodeType[os.environ.get('code_type')]
    # 根据monthlist和taskid生成month
    task_id = int(os.environ.get('task_id'))
    mode = os.environ.get('mode')
    date, algo_class_list = get_date_by_taskid(date_list, global_algo_class_list, task_id)  # taskid 仅在date维度拆分
    # date, func_list = get_date_func_by_taskid(date_list, global_algo_class_list, task_id) # taskid 在date*fun维度拆分
    print("month, func_list", date, algo_class_list)

    # month传给mon_to_fea
    for algo_class in algo_class_list:
        date_to_fea(code_type, date, algo_class, mode)
