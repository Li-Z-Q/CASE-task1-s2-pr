from tqdm import tqdm
import random
import json
from transformers import BertTokenizer, BertConfig
from tools import result_displayer
from models.bert_base import BaseBert


def test():
    gpu_id = 3
    saved_dir = './saved_model'
    tokenizer = BertTokenizer.from_pretrained(saved_dir)
    config = BertConfig.from_pretrained(saved_dir)
    model = BaseBert.from_pretrained(saved_dir, config=config).cuda(gpu_id)

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
