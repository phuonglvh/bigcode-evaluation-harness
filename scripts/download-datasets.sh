#!/bin/bash

set -euox

DIR=./datasets/MultiPL-E
mkdir -p $DIR

# MultiPL-E
wget https://github.com/nuprl/MultiPL-E/blob/main/prompts/humaneval-py-reworded.jsonl -O $DIR/humaneval-py-reworded.jsonl
wget https://github.com/nuprl/MultiPL-E/blob/main/prompts/humaneval-java-reworded.jsonl -O $DIR/humaneval-java-reworded.jsonl

# humaneval-x
DIR=./datasets/humaneval-x
mkdir $DIR

wget https://github.com/THUDM/CodeGeeX/raw/main/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz -O $DIR/humaneval_python.jsonl.gz && gzip -f -d $DIR/humaneval_python.jsonl.gz

wget https://github.com/THUDM/CodeGeeX/raw/main/codegeex/benchmark/humaneval-x/java/data/humaneval_java.jsonl.gz -O $DIR/humaneval_java.jsonl.gz && gzip -f -d $DIR/humaneval_java.jsonl.gz