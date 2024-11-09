#!/bin/bash

set -euox

export EVAL_JAVA_EXTRA_CLASSPATH_FOLDER=$PWD/build/java-bin

# Translate code2code
# py to java
source_generations_path="$(realpath .)/benchmark/CodeLlama-13b-Python-hf/py/improve/pass@1/t0.2-p0.95-k0-batch1-n1/CodeLlama-13b-Python-hf-temp0.2-p0.95-k0-bf16-n1-seed5-batch1-maxlen1024-py-generations-0-158_multiple-py.json"
source_lang=py
source_n_samples=1

AUTHOR="THUDM"
MODEL_NAME="codegeex2-6b"

do_sample=False

max_length=1024
n_samples=2
seed=0
precision=bf16
lang=java

num_return_sequences=1
batch_size=$num_return_sequences

limit_start=0
limit=158
eval_limit_start=0
eval_limit=158

save_every_k_tasks=$source_n_samples
save_every_k_iterations=$((save_every_k_tasks * n_samples / batch_size))

common_name="$MODEL_NAME-do_sample$do_sample-$precision-n$n_samples-seed$seed-batch$batch_size-maxlen$max_length-$lang"

generations_name="$common_name-generations-${limit_start}-${limit}_multiple-$lang"

BASE_DIR=./benchmark/$MODEL_NAME/code2code/$source_lang-$lang/do_sample$do_sample

mkdir -p $BASE_DIR
# rm -rf /tmp/* /var/tmp/*.json
# rm -rf /var/folders/**/*.json

python code_to_code_trans.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks code2code-multiple-java \
    --max_length_generation $max_length \
    --do_sample $do_sample \
    --seed $seed \
    --n_samples $n_samples \
    --batch_size $batch_size \
    --precision $precision \
    --allow_code_execution \
    --trust_remote_code \
    --save_every_k_tasks $save_every_k_iterations \
    --limit_start $limit_start \
    --limit $limit \
    --save_generations \
    --save_generations_path "$BASE_DIR/$common_name-generations-${limit_start}-${limit}.json" \
    --save_references \
    --save_references_path "$BASE_DIR/$common_name-references-${limit_start}-${limit}.json" \
    --source_generations_path "$source_generations_path" \
    --source_lang $source_lang \
    --load_generations_path "/Users/phuonglvh/projects/2170558-thesis-automatic-code-generation-using-machine-learning/bigcode-evaluation-harness/codegeex-chat-pro-0-158_code2code-multiple-java-translated-codes.json" \
    --metric_output_path "$BASE_DIR/$generations_name-eval-${eval_limit_start}-${eval_limit}-evaluation_results.json" \
    --max_memory_per_gpu auto
