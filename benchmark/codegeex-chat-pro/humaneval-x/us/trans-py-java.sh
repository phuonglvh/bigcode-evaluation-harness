#!/bin/bash

set -euox

export EVAL_JAVA_EXTRA_CLASSPATH_FOLDER=$PWD/build/java-bin

AUTHOR="THUDM"
MODEL_NAME="codegeex-chat-pro"

# Translate code2code
# py to java
# prompt_version='v1001'
prompt_version='v1002'
# prompt_version='v1003'
# prompt_version='v1004'
# prompt_version='v1005'
# prompt_version='v1006'
# prompt_version='v1007'
# prompt_version='v1008'
# prompt_version='v1009'

source_generations_path="$(realpath .)/benchmark/$MODEL_NAME/humaneval-x/us/$MODEL_NAME-humaneval_python_java_prompts_$prompt_version-translations-0-164.json"
num_source_generations=164
# source_lang=py


# lang=java
eval_limit_start=0
eval_limit=$num_source_generations


filename=$(basename -- "$source_generations_path")
# extension="${filename##*.}"
generations_name="${filename%.*}"

BASE_DIR=./benchmark/$MODEL_NAME/humaneval-x/us

mkdir -p $BASE_DIR
# rm -rf /tmp/* /var/tmp/*.json
rm -rf /var/folders/**/**.json

python main.py --model "$AUTHOR/$MODEL_NAME" \
    --tasks humanevalx-java \
    --allow_code_execution \
    --trust_remote_code \
    --limit_start $eval_limit_start \
    --limit $eval_limit \
    --load_generations_path "$source_generations_path" \
    --metric_output_path "$BASE_DIR/$generations_name-eval-$eval_limit_start-$eval_limit-evaluation_results.json"
