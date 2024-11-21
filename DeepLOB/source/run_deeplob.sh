#!/bin/bash

# @Author  : Karry Ren
# @Time    : 2024/11/21 16:39
# @Comment : This is the entrance for running deeplob.
#            There might be some different operations to run.
#            But remember, you need to run this by `./DeepLOB/source/run_deeplob task_id`, the task_id is necessary !

# ---- Define the `shell_path` to run (you can change it) ---- #
shell_path="DeepLOB/source/train_test_deeplob.sh"

# ---- Change file of shell_path permission --- #
chmod 755 $shell_path

# ---- Start run the deeplob pipline --- #
if [ $# -gt 0 ]; then # `./source/run.sh xxx` situation
    echo "** You are running: '$shell_path', with task_id: '$1'"
    export task_id=$1
    $shell_path "$task_id"
else
    echo "** ERROR: you need to run this by './DeepLOB/source/run_deeplob task_id', you have no task_id !"
fi
