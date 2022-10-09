import json

def phrase_statistics(target_file):
    with open(target_file, mode='rt', encoding='utf-8') as inp:
        lines = inp.readlines()
        all_phrases = list()
        for line in lines:
            obj = json.loads(line.strip())
            labels = obj['label']['financial_entity']
            for financial_entity in labels:
                all_phrases.append(financial_entity)
        infos = {}
        for phrase in all_phrases:
            phrase_len = len(phrase)
            if phrase_len not in infos:
                infos[phrase_len] = 1
            else:
                infos[phrase_len] += 1
        print('before merge:', len(all_phrases))
        all_phrases = list(set(all_phrases))
        print('---------------------------')
        print(target_file, len(all_phrases))
        print(infos)
        all_splits = {}
        for info in infos:
            num = (info+2)//3
            if num not in all_splits:
                all_splits[num] = infos[info]
            else:
                all_splits[num] += infos[info]
        print("分段区间统计：",all_splits)

if __name__ == '__main__':
    target_train = './dataset_final/train.json'
    phrase_statistics(target_train)

    target_test = './dataset_final/test.json'
    phrase_statistics(target_test)
