import torch
from tqdm import tqdm
import random
import json
from transformers import XLMRobertaTokenizer, XLMRobertaConfig
from tools import result_displayer
from models.bert_base import Model


def test():
    gpu_id = 8

    model_list = []
    tokenizer_list = []
    model_dir_list = [
        './saved_models/IBM/multi_False_1234',
        './saved_models/IBM/multi_False_1235',
        # './saved_models/IBM/multi_False_1236',
        # './saved_models/IBM/multi_False_1237',
        './saved_models/IBM/multi_False_1238'
    ]
    for model_dir in model_dir_list:
        print("model_dir: ", model_dir)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
        config = XLMRobertaConfig.from_pretrained(model_dir)
        model = Model.from_pretrained(model_dir, config=config).cuda(gpu_id)
        model.eval()
        model_list.append(model)
        tokenizer_list.append(tokenizer)

    gold_labels = []
    pre_labels = []
    datas = json.load(open('./data/subtask2-sentence/en-train-dev.json'))

    for data in tqdm(datas):
        with torch.no_grad():
            text = data['text']
            gold_labels.append(data['label'])
            pre_label_list = []
            for model, tokenizer in zip(model_list, tokenizer_list):
                pre_label = model.predict(text, tokenizer, gpu_id)
                pre_label_list.append(pre_label)
            if sum(pre_label_list) > len(pre_label_list) / 2:
                pre_labels.append(1)
            else:
                pre_labels.append(0)
    result_displayer.display_result(gold_labels, pre_labels)

    f = open('./result.json', 'w')
    for l in open('./test_data/english/subtask2-Sentence/test.json'):
        with torch.no_grad():
            l = json.loads(l)
            result = {}
            result['id'] = l['id']
            text = l['sentence']
            pre_label_list = []
            for model, tokenizer in zip(model_list, tokenizer_list):
                pre_label = model.predict(text, tokenizer, gpu_id)
                pre_label_list.append(pre_label)
            if sum(pre_label_list) > len(pre_label_list) / 2:
                result['prediction'] = 1
            else:
                result['prediction'] = 0
        f.write(json.dumps(result) + '\n')

if __name__ == '__main__':
    test()
