#!/bin/bash

# You can set the parameters to any values you want, or simply remove the ones you don't want to set (i.e., run with default parameters)
validation_file="example/valid.csv"
prediction_result_file="output/predictions.csv"
model_name_or_path="prot_bert"
max_length=10
per_device_eval_batch_size=4

python run_imm_classification.py \
    --train_file ${validation_file} \
    --validation_file ${validation_file} \
    --only_evaluation \
    --prediction_result_file ${prediction_result_file} \
    --model_name_or_path ${model_name_or_path} \
    --pad_to_max_length \
    --max_length ${max_length} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \

