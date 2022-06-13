from tqdm import tqdm
import random
import json
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from tools import result_displayer
from models import bert, roberta, multi_lingle


def test():
    gpu_id = 3

    model_path = './saved_models/bert'
    tokenizer_0 = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model_0 = bert.Model.from_pretrained(model_path, config=config).cuda(gpu_id)

    model_path = './saved_models/multi_lingle'
    tokenizer_1 = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model_1 = multi_lingle.Model.from_pretrained(model_path, config=config).cuda(gpu_id)

    model_path = './saved_models/roberta'
    tokenizer_2 = RobertaTokenizer.from_pretrained(model_path)
    config = RobertaConfig.from_pretrained(model_path)
    model_2 = roberta.Model.from_pretrained(model_path, config=config).cuda(gpu_id)

    gold_labels = []
    pre_labels = []
    datas = json.load(open('./data/subtask2-sentence/en-train-dev.json'))
    for data in tqdm(datas):
        text = data['text']
        gold_labels.append(data['label'])

        pre_label_0 = model_0.predict(text, tokenizer_0, gpu_id)
        pre_label_1 = model_1.predict(text, tokenizer_1, gpu_id)
        pre_label_2 = model_2.predict(text, tokenizer_2, gpu_id)

        if pre_label_0 + pre_label_1 + pre_label_2 > 1.5:
            pre_labels.append(1)
        else:
            pre_labels.append(0)

    result_displayer.display_result(gold_labels, pre_labels)

if __name__ == '__main__':
    test()
