#!/bin/bash

# This is the Entrance of Tensor Engineering test.

# ---- Step 1. Construct the `python` environment ---- #
# load the basic anaconda environment
source /opt/share/Modules/init/bash;
module purge;
module load anaconda3-dev;
# load the other environment based on the situation
if [ -d /home/fisher_research/Module ];
then
module load /home/fisher_research/Module/202402/fishermodulefile;
else
module load /mnt/weka/home/test/maming_share/Module/202402/fishermodulefile;
fi

# ---- Step 2. Some OS environment params ---- #
export PYTHONPATH=$(pwd):$PYTHONPATH
export task_id=0

# ---- Step 3. Run test the tensor construction algorithm ---- #
python ./test/test_construction_algo_with_zero_status.py
