#!/bin/bash

set -euox

AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

max_length=1024
temperature=0.8
top_p=0.95
top_k=0

BASE_DIR=./runpod/codegeex2-6b/bugfix-java/bugfix-multiple-java/t$temperature-p$top_p-k$top_k
mkdir -p $BASE_DIR

source_n_samples=200
n_samples=1
seed=0
precision=bf16
lang=java
batch_size=1

eval_limit_start=0
eval_limit=25

limit_start=0
limit=25
gen_limit_start=$((limit_start * source_n_samples))
gen_limit=$((limit * source_n_samples))
source_generations_path='./runpod/codellama-13b-python/java/improve/t0.8-p0.95-k0/CodeLlama-13b-Python-hf-temp0.8-p0.95-k0-bf16-n200-batch10-maxlen1024-java-generations-0-50_multiple-java.json'
# source_generations_path='./runpod/codegeex2-6b/bugfix-java/test-java-gens.json'

NUM_RETURN_SEQUENCES_PER_PROMPT=1
batch_size=$NUM_RETURN_SEQUENCES_PER_PROMPT

save_every_k_tasks=$source_n_samples
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

common_name="$MODEL_NAME-temp$temperature-p$top_p-k$top_k-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations-${limit_start}-${limit}_bugfix-multiple-$lang"
generations_path="$BASE_DIR/$generations_name.json"

python bugfix_main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks bugfix-multiple-$lang \
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
    --save_generations_path "$BASE_DIR/$common_name-generations-$limit_start-$limit.json" \
    --save_references \
    --generation_only \
    --limit_start $gen_limit_start \
    --limit $gen_limit \
    --source_generations_path "$source_generations_path"

flatten_generations_path="$BASE_DIR/$generations_name-flatten.json"
mv $generations_path $flatten_generations_path

# convert flatten generations to [task_gens]
python_concat_script="""
import json
source_n_samples=$source_n_samples
flat_all_gens = json.load(open('$flatten_generations_path', 'r'))
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

rm -rf /tmp/* /var/tmp/*

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks multiple-$lang \
    --max_length_generation $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --top_k $top_k \
    --seed $seed \
    --n_samples $source_n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --limit_start $eval_limit_start \
    --limit $eval_limit \
    --save_every_k_tasks $save_every_k_iterations \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json"
