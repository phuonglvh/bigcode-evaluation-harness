#!/bin/bash

huggingface-cli login

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=10 # 20 if gpu > 16GB, 10 if GPU = 16GB
seed=0
precision=bf16
lang=java
limit_start=0
limit=50
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

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
    --save_generations_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-$limit_start-$limit.json" \
    --save_references \
    --generation_only \
    --token \
    --limit_start $limit_start \
    --limit $limit

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=10 # 20 if gpu > 16GB, 10 if GPU = 16GB
seed=0
precision=bf16
lang=java
limit_start=50
limit=100
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

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
    --save_generations_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-$limit_start-$limit.json" \
    --save_references \
    --generation_only \
    --token \
    --limit_start $limit_start \
    --limit $limit

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=10 # 20 if gpu > 16GB, 10 if GPU = 16GB
seed=0
precision=bf16
lang=java
limit_start=100
limit=100
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

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
    --save_generations_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-$limit_start-$limit.json" \
    --save_references \
    --generation_only \
    --token \
    --limit_start $limit_start \
    --limit $limit

# FULL
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=0
precision=bf16
lang=java
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

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
    --save_generations_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations.json" \
    --save_references \
    --generation_only \
    --token \
    --load_generations_intermediate_paths "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch10-maxlen$max_length-$lang-generations_multiple-java_intermediate.json"
