#!/bin/bash

set -euox

BASE_DIR="${BASE_DIR:-.}"

# Translate code2code
# py to java
source_generations_path=""
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=1 # always fixed to 1
seed=0
precision=bf16
lang=java
source_lang=py
source_n_samples=200
NUM_RETURN_SEQUENCES_PER_PROMPT=1
batch_size=$NUM_RETURN_SEQUENCES_PER_PROMPT

save_every_k_tasks=$source_n_samples
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

python code_to_code_trans.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks code2code-multiple-java \
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
    --source_generations_path "$source_generations_path" \
    --source_lang $source_lang

# java to py
source_generations_path=""
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=1 # always fixed to 1
seed=0
precision=bf16
lang=py
source_lang=java
source_n_samples=200
NUM_RETURN_SEQUENCES_PER_PROMPT=1
batch_size=$NUM_RETURN_SEQUENCES_PER_PROMPT

save_every_k_tasks=$source_n_samples
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

python code_to_code_trans.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks code2code-multiple-py \
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
    --source_generations_path "$source_generations_path" \
    --source_lang $source_lang
