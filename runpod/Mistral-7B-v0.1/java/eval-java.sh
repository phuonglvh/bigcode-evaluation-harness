#!/bin/bash

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
seed=0
precision=bf16
lang=java
full_language=java
# limit_start=0
# limit=164
batch_size=20

save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

common_name="$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations_multiple-$lang"
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
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-evaluation_results.json"

# BLEU score
bleu_predictions_path="$BASE_DIR/$common_name-bleu-predictions_multiple-$lang.txt"
python utils/generations_to_codexglue_bleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$bleu_predictions_path"

bleu_references_path="$BASE_DIR/$common_name-bleu-references_multiple-$lang.jsonl"
python utils/human_eval_x_to_codexglue_bleu.py \
    --language $full_language \
    --load_generations_path "$generations_path" \
    --save_references_path "$bleu_references_path"
(
    cd ./CodeXGLUE/Text-Code/text-to-code || ! echo "cd failure"
    python evaluator/evaluator.py --answers "$bleu_references_path" --predictions "$bleu_predictions_path"
)

# CodeBLEU score
codebleu_predictions_path="$BASE_DIR/$common_name-codebleu-predictions_multiple-$lang.txt"
python utils/generations_to_codexglue_codebleu.py \
    --load_generations_path "$generations_path" \
    --save_predictions_format_path "$codebleu_predictions_path"

codebleu_references_path="$BASE_DIR/$common_name-codebleu-references_multiple-$lang.txt"
python utils/human_eval_x_to_codexglue_codebleu.py \
    --language $full_language \
    --load_generations_path "$generations_path" \
    --save_references_path "$codebleu_references_path"

(
    cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU || ! echo "cd failure"
    python calc_code_bleu.py \
        --lang $full_language \
        --params "0.25,0.25,0.25,0.25" \
        --refs "$codebleu_references_path" \
        --hyp "$codebleu_predictions_path"
)
