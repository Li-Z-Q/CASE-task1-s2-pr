from args import args

import torch
import torch.nn as nn
from models.bert_base import BaseBert
from transformers import BertTokenizer, BertConfig
from tools import data_reader, random_setter, result_displayer
from transformers import AdamW, get_linear_schedule_with_warmup

def dev(model, dev_dataloader):
    model.eval()
    pre_labels = []
    gold_labels = []
    for batch in dev_dataloader:
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0].cuda(args.gpu_id),
                'attention_mask': batch[1].cuda(args.gpu_id),
                'token_type_ids': batch[2].cuda(args.gpu_id),
                'labels': batch[3].cuda(args.gpu_id)
            }
            output = model.forward(**inputs)

        gold_labels += batch[3].cpu().tolist()
        pre_labels += torch.argmax(output['logits'], dim=1).cpu().tolist()
    assert len(gold_labels) == len(pre_labels)

    result = result_displayer.display_result(gold_labels, pre_labels)
    return result

if __name__ == '__main__':

    random_setter.set_random()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    config = BertConfig.from_pretrained(args.pretrained_model, num_labels=2)
    model = BaseBert.from_pretrained(pretrained_model_name_or_path=args.pretrained_model, config=config).cuda(args.gpu_id)
    print('\n')  # always print log

    train_dataloader, dev_dataloader = data_reader.read_data(tokenizer=tokenizer)
    total_step = int(len(train_dataloader) * args.epochs // args.gradient_accumulation_steps)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_ratio * total_step, num_training_steps=total_step) if args.warm_ratio < 1 else None

    # train
    print('\ntrain')
    global_step = 0
    best_dev_result = 0

    model.zero_grad()
    for e in range(args.epochs):
        for i, batch in enumerate(train_dataloader):
            model.train()
            inputs = {
                'input_ids': batch[0].cuda(args.gpu_id),
                'attention_mask': batch[1].cuda(args.gpu_id),
                'token_type_ids': batch[2].cuda(args.gpu_id),
                'labels': batch[3].cuda(args.gpu_id)
            }
            outputs = model.forward(**inputs)
            loss = outputs['loss']
            assert outputs['logits'].shape[-1] == 2  # labels_num = 2

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step() if args.warm_ratio < 1 else None
                model.zero_grad()

                # global_step += 1
                # print("------------------ step: {0}/{1}".format(global_step, total_step))

        dev_result = dev(model, dev_dataloader)
        save_dir = './saved_models/' + args.method
        if dev_result['f'] > best_dev_result:
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            config.save_pretrained(save_dir)
            best_dev_result = dev_result['f']

    print("\n\nbest_dev_result: ", best_dev_result)
    print('done !')