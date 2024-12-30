#!/bin/bash

set -euox

export EVAL_JAVA_EXTRA_CLASSPATH_FOLDER=$PWD/build/java-bin

AUTHOR="OpenAI"
MODEL_NAME="gpt-4o"

# Translate code2code
# py to java

# prompt_version='vRULE-COMB-001-gpt'
# prompt_version='vRULE-COMB-002-gpt'
# prompt_version='vRULE-COMB-003-gpt'
# prompt_version='vRULE-COMB-004-gpt'
# prompt_version='vRULE-COMB-005-gpt'
# prompt_version='vRULE-COMB-006-gpt'
prompt_version='vRULE-COMB-007-gpt'
# prompt_version='vRULE-COMB-008-gpt'
# prompt_version='vRULE-COMB-009-gpt'
# prompt_version='vRULE-COMB-010-gpt'
# prompt_version='vRULE-COMB-011-gpt'
# prompt_version='vRULE-COMB-012-gpt'
# prompt_version='vRULE-COMB-013-gpt'
# prompt_version='vRULE-COMB-014-gpt'
# prompt_version='vRULE-COMB-015-gpt'
# prompt_version='vRULE-COMB-016-gpt'
# prompt_version='vRULE-COMB-017-gpt'
# prompt_version='vRULE-COMB-018-gpt'
# prompt_version='vRULE-COMB-019-gpt'

num_source_generations=164
source_generations_path="$(realpath .)/benchmark/$MODEL_NAME/humaneval-x/us/$MODEL_NAME-humaneval_python_java_prompts_$prompt_version-translations-0-${num_source_generations}.json"
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
