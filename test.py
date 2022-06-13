from tqdm import tqdm
import random
import json
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from tools import result_displayer
from models.bert import BaseBert


def test():
    gpu_id = 3

    model_path = './saved_models/bert'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model_0 = BaseBert.from_pretrained(model_path, config=config).cuda(gpu_id)

    model_path = './saved_models/multi_lingle'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model_1 = BaseBert.from_pretrained(model_path, config=config).cuda(gpu_id)

    model_path = './saved_models/roberta'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model_1 = BaseBert.from_pretrained(model_path, config=config).cuda(gpu_id)

    texts = []
    gold_labels = []
    f = open('/data/subtask2-sentence/en-train.json')
    for l in f.readlines():
        l = json.loads(l)
        texts.append(l['sentence'])
        gold_labels.append(l['label'])

    random.seed(1234)
    random.shuffle(texts)
    random.seed(1234)
    random.shuffle(gold_labels)

    pre_labels = []
    gold_labels = gold_labels[:int(0.2 * len(gold_labels))]
    for text in tqdm(texts[:int(0.2 * len(texts))]):
        pre_labels.append(model.predict(text, tokenizer, gpu_id))

    result_displayer.display_result(gold_labels, pre_labels)

if __name__ == '__main__':
    test()
