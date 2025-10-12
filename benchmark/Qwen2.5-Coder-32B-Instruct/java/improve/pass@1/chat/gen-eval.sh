#!/bin/bash

set -euox

AUTHOR="Qwen"
MODEL_NAME="Qwen2.5-Coder-32B-Instruct"

max_length=1024
do_sample=False
num_return_sequences=1
batch_size=$num_return_sequences

n_samples=1 # pass@1 only
precision=bf16
lang=java

limit_start=0
limit=164
eval_limit_start=0
eval_limit=164

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

# seed=0
# seed=5
seed=10
# seed=15
# seed=20

common_name="$MODEL_NAME-do_sample$do_sample-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_humanevalx-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample

mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks humanevalx-$lang \
    --max_length_generation $max_length \
    --do_sample $do_sample \ \
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
