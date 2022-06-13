from transformers import BertTokenizer, BertForMaskedLM
import torch
import json
import random
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup


def do_tokenize(raw_datas, tokenizer):
    inputs = tokenizer(raw_datas['text'], padding='max_length', max_length=64, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    labels = tokenizer(raw_datas['label'], padding='max_length', max_length=64, return_tensors='pt', truncation=True)["input_ids"]
    labels = torch.where(input_ids == tokenizer.mask_token_id, labels, -100)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    return dataset

def add_mask(datas):
    for i in range(len(datas['text'])):
        new_text = ""
        word_list = datas['text'][i].split()
        for j in range(len(word_list)):
            if random.randint(0, 1000) < 150:
                new_text += '[MASK] '
            else:
                new_text += (word_list[j] + ' ')

        datas['text'][i] = new_text
    return datas

if __name__ == '__main__':

    gpu_id = 2

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").cuda(gpu_id)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_raw_datas = json.load(open('../data/subtask2-sentence/en-train-train.json'))
    random.shuffle(train_raw_datas)
    train_raw_datas = {
        'text': [d['text'] for d in train_raw_datas],
        'label': [d['text'] for d in train_raw_datas]
    }
    train_raw_datas = add_mask(train_raw_datas)

    train_dataset = do_tokenize(train_raw_datas, tokenizer)
    print("do tokenize done, len(train_dataset): ", len(train_dataset))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)

    model.train()
    for e in range(5):
        for batch in train_dataloader:
            inputs = {
                'input_ids': batch[0].cuda(gpu_id),
                'attention_mask': batch[1].cuda(gpu_id),
                'token_type_ids': batch[2].cuda(gpu_id)
            }

            outputs = model(**inputs, labels=batch[3].cuda(gpu_id))

            loss = outputs.loss

            loss.backward()
            optimizer.step()
            model.zero_grad()

    model.save_pretrained('../saved_model/re_pretrain/re_pretrain_model')
    print('save done !')
