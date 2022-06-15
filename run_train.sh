#!/usr/bin/env bash
\
source activate
conda activate py38
\
export method=re_pretrain
export re_pretrain_model=verb
\
export use_cache=False
export cache_dir=./data/subtask2-sentence/cache_bert_base
export train_data_path=./data/subtask2-sentence/en-train-train.json
export dev_data_path=./data/subtask2-sentence/en-train-dev.json
\
export pretrained_model=./saved_models/re_pretrain/${re_pretrain_model}
\
export gpu_id=8
export random_seed=1234
\
export max_seq_len=64
export dev_batch_size=16
export train_batch_size=32
\
export lr=1e-5
export epochs=3
export weight_decay=1e-4
\
export warm_ratio=1.0
export max_grad_norm=99999
export gradient_accumulation_steps=1
\
nohup python -u train.py \
  -method ${method} \
  \
  -use_cache ${use_cache} \
  -cache_dir ${cache_dir} \
  -train_data_path ${train_data_path} \
  -dev_data_path ${dev_data_path} \
  \
  -pretrained_model ${pretrained_model} \
  \
  -gpu_id ${gpu_id} \
  -random_seed ${random_seed} \
  \
  -max_seq_len ${max_seq_len} \
  -dev_batch_size ${dev_batch_size} \
  -train_batch_size ${train_batch_size} \
  \
  -lr ${lr} \
  -epochs ${epochs} \
  -weight_decay ${weight_decay} \
  \
  -warm_ratio ${warm_ratio} \
  -max_grad_norm ${max_grad_norm} \
  -gradient_accumulation_steps ${gradient_accumulation_steps} \
  \
  > ./logs/${method}/log_re_pretrain_model_${re_pretrain_model}.txt &