#!/bin/bash

AUTHOR="codellama"
MODEL_NAME="CodeLlama-13b-Python-hf"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
seed=0
precision=bf16
lang=java
limit_start=0
limit=158
batch_size=10

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
    --save_every_k_tasks 10 \
    --save_generations \
    --save_generations_path "./$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-$limit_start-$limit.json" \
    --save_references \
    --generation_only \
    --limit_start $limit_start \
    --limit $limit \
    --load_generations_path "./$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json" \
    --metric_output_path "./$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang-evaluation_results.json"