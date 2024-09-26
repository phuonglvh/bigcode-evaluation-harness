#!/bin/bash

set -euox

AUTHOR="codellama"
MODEL_NAME="CodeLlama-13b-Python-hf"
max_length=1024

do_sample=False
num_return_sequences=1
batch_size=$num_return_sequences

BASE_DIR=./runpod/codellama-13b-python/java/improve/do_sample$do_sample
mkdir -p $BASE_DIR

n_samples=1
seed=10
precision=bf16
lang=py

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

common_name="$MODEL_NAME-do_sample$do_sample-$precision-n$n_samples-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"
generations_path="$BASE_DIR/$generations_name.json"

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --limit_start $limit_start \
    --limit $limit \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto

rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --limit_start $eval_limit_start \
    --limit $eval_limit \
    --save_every_k_tasks $save_every_k_iterations \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json"
