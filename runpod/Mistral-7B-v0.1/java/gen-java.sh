#!/bin/bash

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=0
precision=bf16
lang=java
limit_start=0
limit=50
save_every_k_tasks=5 # after completing 5 dataset's tasks
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

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
    --limit_start $limit_start \
    --limit $limit

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=0
precision=bf16
lang=java
limit_start=50
limit=50
save_every_k_tasks=5
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

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
    --limit_start $limit_start \
    --limit $limit

BASE_DIR="${BASE_DIR:-.}"
AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=0
precision=bf16
lang=java
limit_start=100
limit=100
save_every_k_tasks=5
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

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
    --limit_start $limit_start \
    --limit $


# concat parts
AUTHOR="mistralai"
MODEL_NAME="Mistral-7B-v0.1"
max_length=1024
temperature=0.8
top_p=0.95
top_k=0
n_samples=200
batch_size=20
seed=0
precision=bf16
lang=java

part_1_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-0-50_multiple-$lang.json"
part_2_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-50-50_multiple-$lang.json"
part_3_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations-100-100_multiple-$lang.json"
full_gen_path="$BASE_DIR/$MODEL_NAME-temp$temperature-p$top_p-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang-generations_multiple-$lang.json"

python_concat_script="
part_1_path = '$part_1_path'
part_2_path = '$part_2_path'
part_3_path = '$part_3_path'
full_gen_path = '$full_gen_path'
import json
part_1_data = json.load(open(part_1_path, 'r'))[0:50]
part_2_data = json.load(open(part_2_path, 'r'))[0:50]
part_3_data = json.load(open(part_3_path, 'r'))[0:61]
all_gens = part_1_data + part_2_data + part_3_data
json.dump(all_gens, open(full_gen_path, 'w'))
print(f'wrote {len(all_gens)} to \"{full_gen_path}\"')
"

python -c "$python_concat_script"