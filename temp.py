from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
import torch

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
# config.hidden_size = 1024
config.num_attention_heads = 16
config.num_hidden_layers = 24
model = XLMRobertaModel.from_pretrained("xlm-roberta-base", config=config)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

a = torch.load('xlm-roberta-base/pytorch_model.bin')
print(a)