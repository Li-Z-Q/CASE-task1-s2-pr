import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertConfig, BertModel, BertForTokenClassification

class Model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config=config)
        self.linear = nn.Linear(config.hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduction='sum')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sentence_embedding = outputs.pooler_output
        logits = self.log_softmax(self.linear(sentence_embedding))
        loss = self.nllloss(logits, labels)

        outputs['LZQ'] = 'LZQ'
        outputs['loss'] = loss
        outputs['logits'] = logits

        return outputs

    def predict(self, text, tokenizer, gpu_id):
        inputs = tokenizer(text, padding='max_length', max_length=64, return_tensors='pt', truncation=True)
        inputs['labels'] = torch.tensor([0]).cuda(gpu_id)
        inputs['input_ids'] = inputs['input_ids'].cuda(gpu_id)
        inputs['attention_mask'] = inputs['attention_mask'].cuda(gpu_id)
        inputs['token_type_ids'] = inputs['token_type_ids'].cuda(gpu_id)

        outputs = self.forward(**inputs)

        pre_label = torch.argmax(outputs.logits).cpu()
        return pre_label
