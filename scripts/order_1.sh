#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-2024}"
BASE_MODEL="${BASE_MODEL:-path_t5}"
DATA_DIR="${DATA_DIR:-../CL_Benchmark}"

deepspeed --master_port 20028 src/run_householder_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$BASE_MODEL" \
   --data_dir "$DATA_DIR" \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-02 \
   --num_train_epochs 1 \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --seed "$SEED"


sleep 5

deepspeed --master_port 20028 src/run_householder_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/1-dbpedia/adapter \
   --data_dir "$DATA_DIR" \
   --task_config_dir configs/order1_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-02 \
   --num_train_epochs 1 \
   --run_name order1_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --seed "$SEED"

sleep 5

deepspeed --master_port 20028 src/run_householder_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/2-amazon/adapter \
   --data_dir "$DATA_DIR" \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/3-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-02 \
   --num_train_epochs 1 \
   --run_name order1_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --seed "$SEED"

sleep 5

deepspeed --master_port 20028 src/run_householder_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_1/3-yahoo/adapter \
   --data_dir "$DATA_DIR" \
   --task_config_dir configs/order1_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_1/4-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 2 \
   --learning_rate 3e-02 \
   --num_train_epochs 1 \
   --run_name order1_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --seed "$SEED"
