import json
import stanza

train_raw_datas = json.load(open('./data/subtask2-sentence/en-train-train.json'))
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, ner', use_gpu=True)
texts = ['aaa sss sss', 'aaa sss sss']
documents = stanza_nlp(texts)
print(documents)
