import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-method',default='xlm_roberta', type=str)

parser.add_argument('-multi', default='False', type=str)
parser.add_argument('-random_seed', default=1234, type=int)

parser.add_argument('-use_cache', default='False', type=str)
parser.add_argument('-cache_dir', default='./data/subtask2-sentence/cache_xlm_roberta_large_base')
parser.add_argument('-train_data_path', default='./data/subtask2-sentence/en-train-train.json', type=str)
parser.add_argument('-dev_data_path', default='./data/subtask2-sentence/en-train-dev.json', type=str)

parser.add_argument('-pretrained_model', default='xlm-roberta-large', type=str)

parser.add_argument('-gpu_id', default=5, type=int)

parser.add_argument('-max_seq_len', default=64, type=int)
parser.add_argument('-dev_batch_size', default=16, type=int)
parser.add_argument('-train_batch_size', default=32, type=int)

parser.add_argument('-lr', default=2e-5, type=float)
parser.add_argument('-epochs', default=3, type=int)
parser.add_argument('-weight_decay', default=0.0001, type=float)

parser.add_argument('-warm_ratio', default=1.0, type=float)
parser.add_argument('-max_grad_norm', default=99999, type=float)
parser.add_argument('-gradient_accumulation_steps', default=1, type=int)

parser.add_argument('-multi_ling_1_path', default='./data/subtask2-sentence/es-train.json', type=str)
parser.add_argument('-multi_ling_2_path', default='./data/subtask2-sentence/pr-train.json', type=str)


args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
print()