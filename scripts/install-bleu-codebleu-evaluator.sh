#!/bin/bash

set -euox

git clone https://github.com/phuonglvh/CodeXGLUE.git
(cd ./CodeXGLUE && git checkout feature/thesis)

(
    cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU
    pip install tree_sitter
    cd parser && bash build.sh
)