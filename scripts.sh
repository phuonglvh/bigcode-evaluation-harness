#!/bin/sh

python main.py --model codeparrot/codeparrot-small \
    --tasks multiple-py \
    --max_length_generation 512 \
    --temperature 0.5 \
    --n_samples 10 \
    --batch_size 10 \
    --allow_code_execution \
    --limit 2 \
    --generation_only \
    --save_generations \
    --save_generations_path generations_py.json

python main.py --model codeparrot/codeparrot-small \
    --tasks multiple-java \
    --max_length_generation 512 \
    --temperature 0.5 \
    --n_samples 10 \
    --batch_size 10 \
    --allow_code_execution \
    --limit 2 \
    --generation_only \
    --save_generations \
    --save_generations_path generations_java.json

docker run \
    -v "$(pwd)/generations_py.json":/app/generations_py.json:ro \
    -v "$(pwd)/containers/py":/tmp \
    -v "$(pwd)/containers/.cache/huggingface/datasets":/root/.cache/huggingface/datasets \
    --rm \
    -it ghcr.io/bigcode-project/evaluation-harness-multiple python3 main.py \
    --model codeparrot/codeparrot-small \
    --tasks multiple-py \
    --load_generations_path /app/generations_py.json \
    --metric_output_path /tmp/evaluation_results.json \
    --temperature 0.2 \
    --n_samples 10 \
    --batch_size 10 \
    --allow_code_execution \
    --limit 2

docker run \
    -v "$(pwd)/generations_java.json":/app/generations_java.json:ro \
    -v "$(pwd)/containers/java":/tmp \
    -v "$(pwd)/containers/.cache/huggingface/datasets":/root/.cache/huggingface/datasets \
    --rm \
    -it ghcr.io/bigcode-project/evaluation-harness-multiple python3 main.py \
    --model codeparrot/codeparrot-small \
    --tasks multiple-java \
    --load_generations_path /app/generations_java.json \
    --metric_output_path /tmp/evaluation_results.json \
    --temperature 0.2 \
    --n_samples 10 \
    --batch_size 10 \
    --allow_code_execution \
    --limit 2
