#!/bin/bash

# @Author  : Karry Ren
# @Time    : 2024/03/15 10:25
# @Comment : This is the entrance for Construction of Tensor Engineering.
#            Run this by `./run_construction.sh` directly

# ---- Some environment settings ---- #
# shellcheck disable=SC2155
export PYTHONPATH=$(pwd):$PYTHONPATH

# ---- Set the basic os params for following use ---- #
export start_date="20220101"
export end_date="20230101"
export mode="a" # different modes will have different operation
export n_process=1 # if == 1, for debug

# ---- Prepare the tensor groups and code types to do the construction ---- #
#tensor_group_list=("$USER") # the list of tensor group
tensor_group_list=("base") # the list of tensor group
code_type_list=("FINANCIAL_FUTURE") # the list of code type

# ---- For loop the `tensor_group` and `code_type` to construct each factor ---- #
for tensor_group_item in "${tensor_group_list[@]}"
do
    for code_type_item in "${code_type_list[@]}"
    do
        export tensor_group=$tensor_group_item;
        export code_type=$code_type_item;
        echo "***************************************"
        echo "** code_type: '$code_type'"
        echo "** tensor_group: '$tensor_group'"
        echo "** start_date: '$start_date'"
        echo "** end_date: '$end_date'"
        echo "** mode: '$mode'"
        echo "** n_process: '$n_process'"
        echo "***************************************"
        echo "Top is the information from shell settings. ⬆ "
        echo ""
        python3.8 TensorEngineering/main/construction.py
    done
done
