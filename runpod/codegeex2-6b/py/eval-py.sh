#!/bin/bash

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

save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))


# pass@{1,10,100}
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
    --load_generations_path "./$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json" \
    --metric_output_path "./$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang-evaluation_results.json" \
    --use_auth_token

python utils/generations_to_codexglue_codebleu.py \
    --generations_path runpod/$MODEL_NAME/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json \
    --predictions_path ./results/codebleu/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-predictions_multiple-$lang.txt

python utils/generations_to_codexglue_bleu.py \
    --generations_path runpod/$MODEL_NAME/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json \
    --predictions_path ./results/bleu/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-predictions_multiple-$lang.txt

python utils/human_eval_x_to_codexglue_codebleu.py \
    -lang python \
    --predictions_path runpod/$MODEL_NAME/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json \
    --references_path ./datasets/references/codebleu/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-references_multiple-$lang.text

python utils/human_eval_x_to_codexglue_bleu.py \
    -lang python \
    --predictions_path runpod/$MODEL_NAME/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json \
    --references_path ./datasets/references/bleu/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-references_multiple-$lang.jsonl


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