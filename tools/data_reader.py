from args import args

import stanza
import os
import json
import torch
import random
from collections import Counter
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler


def pos_ner_fn(datas):
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, ner', use_gpu=False)
    for i in range(len(datas['text'])):
        new_text = ""

        text = datas['text'][i]
        document = stanza_nlp(text)
        for sentence in document.sentences:
            for word, token in zip(sentence.words, sentence.tokens):
                ner = token.ner
                pos = word.pos

                if ner != 'O':
                    new_text += (word.text + ' ')
                else:
                    new_text += (pos + ' ')

        datas['text'][i] = new_text

    return datas

def do_tokenize(raw_datas, tokenizer):
    inputs = tokenizer(raw_datas['text'], padding='max_length', max_length=args.max_seq_len, return_tensors='pt', truncation=True)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    labels = torch.tensor(raw_datas['label'])

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    return dataset


def read_data(tokenizer):

    if os.path.exists(args.cache_dir) and args.use_cache == 'True':
        (train_dataloader, dev_dataloader) = torch.load(args.cache_dir + '/cache_data.pt')
        print('load data from {} done !'.format(args.cache_dir))

    else:
        train_raw_datas = json.load(open(args.train_data_path))
        random.shuffle(train_raw_datas)
        train_raw_datas = {
            'text': [d['text'] for d in train_raw_datas],
            'label': [d['label'] for d in train_raw_datas]
        }
        train_raw_datas = pos_ner_fn(train_raw_datas)
        print("Counter([d['label'] for d in train_raw_datas]): ", Counter([l for l in train_raw_datas['label']]))

        dev_raw_datas = json.load(open(args.dev_data_path))
        dev_raw_datas = {
            'text': [d['text'] for d in dev_raw_datas],
            'label': [d['label'] for d in dev_raw_datas]
        }

        dev_dataset = do_tokenize(dev_raw_datas, tokenizer)
        print("do tokenize done, len(dev_dataset): ", len(dev_dataset))
        dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.dev_batch_size)

        train_dataset = do_tokenize(train_raw_datas, tokenizer)
        print("do tokenize done, len(train_dataset): ", len(train_dataset))
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)

        os.mkdir(args.cache_dir) if os.path.exists(args.cache_dir) == False else None
        torch.save((train_dataloader, dev_dataloader), args.cache_dir + '/cache_data.pt')
        print('save {} done !'.format(args.cache_dir + '/cache_data.pt'))

    return train_dataloader, dev_dataloader


