#!/usr/bin/env bash

MODELS_FOLDER=$1
BASE_CONFIG=$2
PARAMETER_GRID=$3
GRID_PLAN_FOLDER=$4

python generate_grid_search_plan.py $BASE_CONFIG $PARAMETER_GRID $GRID_PLAN_FOLDER

for config_file in `ls $GRID_PLAN_FOLDER`; do
  model_folder=`basename $config_file .json`
  sbatch train_slurm.sh \
    ../data/babi_task6 \
    ../data/babi_task6_ood_0.2_0.4 \
    $MODELS_FOLDER/$model_folder \
    $GRID_PLAN_FOLDER/$config_file \
    --custom_vocab unified_vocab.txt
done
