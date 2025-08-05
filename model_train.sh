#!/bin/bash

# You can set the parameters to any values you want, or simply remove the ones you don't want to set (i.e., run with default parameters)
train_file="example/train.csv"
validation_file="example/valid.csv"
model_name_or_path="prot_bert"
learning_rate=1e-5
weight_decay=0.0
max_length=10
num_train_epochs=1
per_device_train_batch_size=2
per_device_eval_batch_size=4
output_dir="models"
seed=1

python run_imm_classification.py \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --model_name_or_path ${model_name_or_path} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --pad_to_max_length \
    --max_length ${max_length} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --output_dir ${output_dir} \
    --seed ${seed} \
    --with_tracking

