#!/bin/bash

git clone https://github.com/phuonglvh/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
git checkout feature/thesis
pip install -r requirements.txt
bash scripts/install-useful-tools.sh