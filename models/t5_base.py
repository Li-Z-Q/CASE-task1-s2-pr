import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Model, T5ForConditionalGeneration, T5EncoderModel

class Model(nn.Module):
    def __init__(self, pretrained_model, config):
        super().__init__()
        config.output_hidden_states = True
        self.t5 = T5Model.from_pretrained(pretrained_model, config=config)
        self.linear = nn.Linear(config.hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduction='sum')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            labels=None
    ):

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = last_hidden_state[:, -2, :]
        logits = self.log_softmax(self.linear(sentence_embedding))

        loss = self.nllloss(logits, labels)

        outputs['LZQ'] = 'LZQ'
        outputs['loss'] = loss
        outputs['logits'] = logits

        return outputs

    def predict(self, text, tokenizer, gpu_id):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding='max_length')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        prompt = 'answer is yes or no : <extra_id_0>'
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64, padding='max_length')
        decoder_input_ids = inputs['input_ids']

        inputs = {
            'input_ids': input_ids.cuda(gpu_id),
            'attention_mask': attention_mask.cuda(gpu_id),
            'decoder_input_ids': decoder_input_ids.cuda(gpu_id),
            'labels': torch.tensor(0).cuda(gpu_id)
        }

        outputs = self.forward(**inputs)

        pre_label = torch.argmax(outputs.logits).cpu()
        return pre_label

    def save(self, dir):
        torch.save(self.state_dict(), dir + '/model.pt')

    def load(self, dir):
        self.load_state_dict(torch.load(dir + '/model.pt'))
        return self