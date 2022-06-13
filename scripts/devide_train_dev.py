import json
import random
from collections import Counter


if __name__ == '__main__':
    file_path = '../data/subtask2-sentence/en-train.json'

    all_data = []
    f = open(file_path)
    for l in f.readlines():
        try:
            l = json.loads(l)
        except:
            print("error: ", l)
            break
        data = {'id': l['id'], 'text': l['sentence'], 'label': l['label']}
        all_data.append(data)

    print("file_path: ", file_path)
    print("len(all_data): ", len(all_data))
    print("Counter([d['label'] for d in all_data]): ", Counter([d['label'] for d in all_data]))

    random.seed(1234)
    random.shuffle(all_data)

    dev_data_num = 2282
    print("dev_data_num: ", dev_data_num)
    dev_data = all_data[:dev_data_num]
    train_data = all_data[dev_data_num:]

    json.dump(dev_data, open(file_path[:-5] + '-dev.json', 'w'))
    json.dump(train_data, open(file_path[:-5] + '-train.json', 'w'))

