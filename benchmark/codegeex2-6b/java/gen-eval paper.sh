#!/bin/bash

set -euox

AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

# https://github.com/THUDM/CodeGeeX2/blob/main/scripts/run_humanevalx.sh
# pass@1 greedy
do_sample=False
max_length=1024
temperature=1
top_k=1
top_p=1
num_return_sequences=1
batch_size=$num_return_sequences
n_samples=20 # pass@1 only
seed=42

# pass@1 sampling
do_sample=True
max_length=1024
temperature=0.2
top_k=0
top_p=0.95
num_return_sequences=1
batch_size=$num_return_sequences
n_samples=20
seed=42


precision=bf16
lang=java

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

BASE_DIR=./benchmark/$MODEL_NAME/$lang/improve/pass@1/t$temperature-p$top_p-k$top_k-batch$batch_size-n$n_samples

mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

common_name="$MODEL_NAME-temp$temperature-p$top_p-k$top_k-$precision-n$n_samples-seed$seed-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --temperature $temperature \
    --top_p $top_p \
    --top_k $top_k \
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

AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
do_sample=False
num_return_sequences=10
batch_size=$num_return_sequences
n_samples=20
seed=42
precision=bf16
lang=java

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

do_sample=False

# 10
num_beams=10
common_name="$MODEL_NAME-do_sample$do_sample-num_beams$num_beams-$precision-n$n_samples-seed$seed-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/$lang/do_sample$do_sample
mkdir -p $BASE_DIR
rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --num_beams $num_beams \
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