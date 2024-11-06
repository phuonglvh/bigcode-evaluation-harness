#!/bin/bash

set -euox

AUTHOR="codellama"
MODEL_NAME="CodeLlama-13b-Python-hf"

max_length=1024
temperature=0.9
top_k=0
top_p=0.9
batch_size=10

BASE_DIR=./benchmark/$MODEL_NAME/$lang/improve/t$temperature-p$top_p-k$top_k-batch$batch_size
mkdir -p $BASE_DIR

n_samples=200
seed=0
precision=bf16
lang=java

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

common_name="$MODEL_NAME-temp$temperature-p$top_p-k$top_k-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"
generations_path="$BASE_DIR/$generations_name.json"

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
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
    --generation_only \
    --limit_start $limit_start \
    --limit $limit \
    --max_memory_per_gpu auto

# use case: load intermediate generations
intermediate_generations_path="$BASE_DIR/$generations_name" 
intermediate_generations_path+="_intermediate.json"

echo $generations_path
echo $intermediate_generations_path
python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
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
    --load_generations_intermediate_paths "$intermediate_generations_path" \
    --save_references \
    --generation_only \
    --limit_start $limit_start \
    --limit $limit \
    --max_memory_per_gpu auto

rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --top_k $top_k \
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
