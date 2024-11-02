#!/bin/bash

set -euox

git clone https://github.com/phuonglvh/CodeXGLUE.git
(cd ./CodeXGLUE && git checkout feature/thesis)

(
    cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU || ! echo "cd failure"
    pip install setuptools tree_sitter==0.21.3
    cd parser && bash build.sh
)