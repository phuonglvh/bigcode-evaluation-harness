#!/bin/bash

huggingface-cli login

BASE_DIR="${BASE_DIR:-.}"

# FULL
AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20 # not used in evaluation mode
seed=0
precision=bf16
lang=java
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

generations_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json"

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
    --token \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang-evaluation_results.json"
    
# generations_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json"

python utils/generations_to_codexglue_codebleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-codebleu-predictions_multiple-$lang.txt"

python utils/generations_to_codexglue_bleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-bleu-predictions_multiple-$lang.txt"

python utils/human_eval_x_to_codexglue_codebleu.py \
    --language java \
    --load_generations_path "$generations_path" \
    --save_references_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-codebleu-references_multiple-$lang.txt"

python utils/human_eval_x_to_codexglue_bleu.py \
    --language java \
    --load_generations_path "$generations_path" \
    --save_references_path "$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-bleu-references_multiple-$lang.jsonl"
