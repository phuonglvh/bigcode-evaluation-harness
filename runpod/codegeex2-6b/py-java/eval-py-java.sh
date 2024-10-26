#!/bin/bash

set -euox

BASE_DIR="${BASE_DIR:-.}"

# Translate code2code
# py to java
# source_generations_path="/workspace/bigcode-evaluation-harness/runpod/codellama-13b-python/CodeLlama-13b-Python-hf-temp0.8-p0.95-bf16-n200-batch32-maxlen512-py-generations_multiple-py.json"
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
# source_lang=py
source_n_samples=200

limit_start=$((0 * source_n_samples))
limit=$((40 * source_n_samples))

NUM_RETURN_SEQUENCES_PER_PROMPT=1
batch_size=$NUM_RETURN_SEQUENCES_PER_PROMPT

save_every_k_tasks=$source_n_samples
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

eval_n_samples=$source_n_samples
eval_limit_start=$((limit_start / source_n_samples))
eval_limit=$((limit / source_n_samples))

common_name="$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_code2code-multiple-$lang"
generations_path="$BASE_DIR/$generations_name.json"

# convert flatten generations to [task_gens]
python_concat_script="""
import json
source_n_samples=$source_n_samples
flat_all_gens = json.load(open('$BASE_DIR/$generations_name-flatten.json', 'r'))
num_iters = int(len(flat_all_gens)/source_n_samples)

all_gens = []
for i in range(num_iters):
    task_gens = flat_all_gens[i*source_n_samples:(i+1)*source_n_samples]
    task_gens = [gens[0] for gens in task_gens]
    all_gens.append(task_gens)

json.dump(all_gens, open('$generations_path', 'w'))
print('saved merged gens at \"$generations_path\"')
"""

python -c "$python_concat_script"

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --top_k $top_k \
    --seed $seed \
    --n_samples $eval_n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --token \
    --limit_start $eval_limit_start \
    --limit $eval_limit \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-evaluation_results.json"
