
#!/bin/bash

set -euox

AUTHOR="codellama"
MODEL_NAME="CodeLlama-13b-Python-hf"
max_length=1024

temperature=0.8
top_p=0.95
top_k=0

BASE_DIR=./runpod/codellama-13b-python/java/improve/t$temperature-p$top_p-k$top_k
mkdir -p $BASE_DIR

n_samples=200
seed=0
precision=bf16
lang=java
batch_size=10

eval_limit_start=0
eval_limit=50

save_every_k_tasks=1 # after completing k dataset's tasks
save_every_k_iterations=$(($save_every_k_tasks*$n_samples/$batch_size))

common_name="$MODEL_NAME-temp$temperature-p$top_p-k$top_k-$precision-n$n_samples-batch$batch_size-maxlen$max_length-$lang"
generations_name="$common_name-generations_multiple-$lang"
generations_path="./runpod/codellama-13b-python/java/CodeLlama-13b-Python-hf-temp0.8-p0.95-bf16-n200-batch10-maxlen1024-java-generations_multiple-java.json"

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
    --limit_start $eval_limit_start \
    --limit $eval_limit \
    --save_every_k_tasks $save_every_k_iterations \
    --load_generations_path "$generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json"