#!/bin/bash

mkdir -p log

echo "========== Training Start =========="
python3 main_mix_2.py 2>&1 | tee log/train_$(date +"2"+"%Y%m%d_%H%M").log

echo "========== Evaluation Start =========="
python3 eval_mix_2.py 2>&1 | tee log/eval_$(date +"2"+"%Y%m%d_%H%M").log