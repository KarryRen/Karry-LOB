#!/bin/bash

# This is the entrance for Construction of Tensor Engineering.
# Run this by `./run_construction.sh` directly

# ---- Step 1. Set the basic os params for following use ---- #
export start_date="20220101"
export end_date="20231231"
export mode="a"
export freq="M"
export n_process=16

# ---- Step 2. Prepare the tensor groups and code types to do the construction ---- #
tensor_group_list=("$USER") # the list of tensor group
code_type_list=("FINANCIAL_FUTURE") # the list of code type

# ---- Step 3. For loop the `tensor_group` and `code_type` to construct each factor ---- #
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
        echo "** freq: '$freq'"
        echo "** n_process: '$n_process'"
        echo "***************************************"
        python3.8 ./main/construction.py
    done
done