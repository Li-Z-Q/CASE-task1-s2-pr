import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-method',default='roberta', type=str)

parser.add_argument('-use_cache', default='False', type=str)
parser.add_argument('-cache_dir', default='./data/subtask2-sentence/cache_bert_base')
parser.add_argument('-train_data_path', default='./data/subtask2-sentence/en-train-train.json', type=str)
parser.add_argument('-dev_data_path', default='./data/subtask2-sentence/en-train-dev.json', type=str)

parser.add_argument('-pretrained_model', default='bert-base-uncased', type=str)

parser.add_argument('-gpu_id', default=0, type=int)
parser.add_argument('-random_seed', default=1234, type=int)

parser.add_argument('-max_seq_len', default=64, type=int)
parser.add_argument('-dev_batch_size', default=16, type=int)
parser.add_argument('-train_batch_size', default=32, type=int)

parser.add_argument('-lr', default=1e-5, type=float)
parser.add_argument('-epochs', default=5, type=int)
parser.add_argument('-weight_decay', default=0.01, type=float)

parser.add_argument('-warm_ratio', default=1.0, type=float)
parser.add_argument('-max_grad_norm', default=99999, type=float)
parser.add_argument('-gradient_accumulation_steps', default=1, type=int)


args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))
print()