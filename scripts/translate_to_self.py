from tqdm import tqdm, trange
import json
from googletrans import Translator


# de ru fr

def translate_chain(text, translator, chain):
    for language in chain:
        text = translator.translate(text, dest=language).text
    return text

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-chain', default='fr_ja_ru_en', type=str)
    args = parser.parse_args()

    chain = args.chain.split('_')
    print("chain: ", chain)

    translator = Translator(service_urls=['translate.google.cn'])

    new_datas = []
    datas = json.load(open('../data/subtask2-sentence/en-train-train.json'))
    for data in tqdm(datas):
        text = data['text']
        label = data['label']

        try:
            new_data = {}
            new_data['label'] = label
            new_data['text'] = translate_chain(text, translator, chain)
            new_datas.append(new_data)
        except:
            print('fail {}'.format(data))

    json.dump(new_datas, open('../data/subtask2-sentence/en-train-train-{}.json'.format(args.chain), 'w'))
