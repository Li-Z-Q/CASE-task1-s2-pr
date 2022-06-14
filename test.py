import torch
from tqdm import tqdm
import random
import json
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, T5Tokenizer, T5Config
from tools import result_displayer
from models import bert, roberta, multi_lingle, t5_encoder, t5


def test():
    gpu_id = 4

    # model_dir = './saved_models/bert'
    # tokenizer_0 = BertTokenizer.from_pretrained(model_dir)
    # config = BertConfig.from_pretrained(model_dir)
    # model_0 = bert.Model.from_pretrained(model_dir, config=config).cuda(gpu_id)
    #
    # model_dir = './saved_models/multi_lingle'
    # tokenizer_1 = BertTokenizer.from_pretrained(model_dir)
    # config = BertConfig.from_pretrained(model_dir)
    # model_1 = multi_lingle.Model.from_pretrained(model_dir, config=config).cuda(gpu_id)
    #
    # model_dir = './saved_models/roberta'
    # tokenizer_2 = RobertaTokenizer.from_pretrained(model_dir)
    # config = RobertaConfig.from_pretrained(model_dir)
    # model_2 = roberta.Model.from_pretrained(model_dir, config=config).cuda(gpu_id)

    # pre_model = 't5-base'
    # model_dir = './saved_models/t5_encoder'
    # tokenizer_3 = T5Tokenizer.from_pretrained(pre_model, model_max_length=512)
    # config = T5Config.from_pretrained(pre_model)
    # model_3 = t5_encoder.Model(pre_model, config=config).load(model_dir).cuda(gpu_id)
    #
    pre_model = 't5-base'
    model_dir = './saved_models/t5'
    tokenizer_4 = T5Tokenizer.from_pretrained(pre_model, model_max_length=512)
    config = T5Config.from_pretrained(pre_model)
    model_4 = t5.Model(None, config=config).load(model_dir).cuda(gpu_id)

    # model_0.eval()
    # model_1.eval()
    # model_2.eval()
    # model_3.eval()
    model_4.eval()

    gold_labels = []
    pre_labels = []
    datas = json.load(open('./data/subtask2-sentence/en-train-dev.json'))

    with torch.no_grad():
        for data in tqdm(datas):
            text = data['text']
            gold_labels.append(data['label'])

            # pre_label_0 = model_0.predict(text, tokenizer_0, gpu_id)
            # pre_label_1 = model_1.predict(text, tokenizer_1, gpu_id)
            # pre_label_2 = model_2.predict(text, tokenizer_2, gpu_id)
            # pre_label_3 = model_3.predict(text, tokenizer_3, gpu_id)
            pre_label_4 = model_4.predict(text, tokenizer_4, gpu_id)

            # if pre_label_0 + pre_label_1 + pre_label_2 > 1.5:
            #     pre_labels.append(1)
            # else:
            #     pre_labels.append(0)

            pre_labels.append(pre_label_4)

    result_displayer.display_result(gold_labels, pre_labels)

if __name__ == '__main__':
    test()
