#!/usr/bin/env bash
\
source activate
conda activate py38
\
export method=IBM
export multi=False
export random_seed=1236
\
export gpu_id=8
\
export use_cache=False
export cache_dir=./data/subtask2-sentence/cache_roberta_base
export train_data_path=./data/subtask2-sentence/en-train-train.json
export dev_data_path=./data/subtask2-sentence/en-train-dev.json
\
export pretrained_model=xlm-roberta-large
\
export max_seq_len=128
export dev_batch_size=16
export train_batch_size=32
\
export lr=2e-5
export epochs=5
export weight_decay=1e-2
\
export warm_ratio=0.8
export max_grad_norm=99999
export gradient_accumulation_steps=1
\
export multi_ling_1_path=./data/subtask2-sentence/es-train.json
export multi_ling_2_path=./data/subtask2-sentence/pr-train.json
\
nohup python -u train.py \
  -method ${method} \
  -multi ${multi} \
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
  > ./logs/${method}/log_multi_${multi}_random_seed_${random_seed}_warm_ratio_${warm_ratio}.txt &