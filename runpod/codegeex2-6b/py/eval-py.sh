#!/bin/bash

BASE_DIR="${BASE_DIR:-.}"

AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=42
precision=bf16
lang=py
full_language=python

save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

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
    --token \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-evaluation_results.json"
    
# generations_path="$BASE_DIR/$generations_name.json"

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
    cd ./CodeXGLUE/Text-Code/text-to-code
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
    cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU
    python calc_code_bleu.py \
        --lang $full_language \
        --params "0.25,0.25,0.25,0.25" \
        --refs "$codebleu_references_path" \
        --hyp "$codebleu_predictions_path"
)


# part_1 = '/workspace/bigcode-evaluation-harness/codegeex2-6b-temp0.8-p0.95-bf16-n200-batch5-maxlen1024-py-generations_0-50-multiple-py.json'
# part_2 = '/workspace/bigcode-evaluation-harness/codegeex2-6b-temp0.8-p0.95-bf16-n200-batch20-maxlen1024-py-generations-50-50_multiple-py.json'
# part_3 = '/workspace/bigcode-evaluation-harness/codegeex2-6b-temp0.8-p0.95-bf16-n200-batch10-maxlen1024-py-generations-100-61_multiple-py.json'
# full = './codegeex2-6b-temp0.8-p0.95-bf16-n200-batch20-maxlen1024-py-generations_multiple-py.json'
# import json
# part_1_data = json.load(open(part_1, 'r'))[0:50]
# part_2_data = json.load(open(part_2, 'r'))[0:50]
# part_3_data = json.load(open(part_3, 'r'))[0:61]
# all_tasks = part_1_data + part_2_data + part_3_data
# json.dump(all_tasks, open(full, 'w'))