#!/usr/bin/env bash

source activate
conda activate py38

export method=normal
export save_dir=../saved_models/re_pretrain/${method}

nohup python -u do_re_pretrain.py -method ${method} > ./log_method_${method}.txt &