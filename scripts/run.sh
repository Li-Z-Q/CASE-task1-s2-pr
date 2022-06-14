#!/usr/bin/env bash

source activate
conda activate py38

export method=noun
export save_dir=../saved_models/re_pretrain/${method}_re_pretrain

nohup python -u do_re_pretrain.py -method ${method} > ./log_method_${method}.txt &