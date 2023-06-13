import json
import random
import tqdm
""" 
train_extra.json 只用于保留 数据增强的扩充训练集
train_new.json 同时用于保留 原训练集 + 数据增强的扩充训练集
train_new_random.json 同时用于保留 原训练集 + 数据增强的扩充训练集 + 已随机打混
"""
wrong_count = 0
extra_tokens = ['1.', '2.', '3.', '4.', '5.', "\""]  # ...
with open('./ChatGPT_responses/answers_ChatGPT.json', mode='rt', encoding='utf-8') as inp, \
    open('./Fintech-Key-Phrase-new/train_extra.json', mode='wt', encoding='utf-8') as oup1, \
    open('./Fintech-Key-Phrase-new/train_new.json', mode='wt', encoding='utf-8') as oup2:
    lines = inp.readlines()
    for line in tqdm.tqdm(lines):
        # index sentence ChatGPT_answers label
        # ==> text, "label": {"financial_entity": {"轮胎模具": [[0, 3]], "硫化机行业": [[18, 22]]}}
        try:
            obj = json.loads(line.strip())
        except Exception as e:
            print(e.args, '选择跳过.')
            wrong_count +=1
            continue
        sentence, ChatGPT_answers, label = obj['sentence'], obj['ChatGPT_answers'], obj['label']
        # print('正在处理:', sentence)

        # 提前存入原训练集 of sentence 至 train_new.json.
        keyphrases_info = {}
        for phrase in label:
            start_idx = sentence.find(phrase)
            collected_info = []
            while start_idx != -1:
                collected_info.append([start_idx, start_idx + len(phrase) - 1])
                start_idx = sentence.find(phrase, start_idx + len(phrase))
            if len(collected_info) > 0:
                keyphrases_info[phrase] = collected_info
        if len(list(keyphrases_info)) != 0:
            json_row = {'text': sentence, 'label': {'financial_entity': keyphrases_info}}
            oup2.write(json.dumps(json_row, ensure_ascii=False) + '\n')
        else:
            print('未识别到keyphrase, 跳过.')
            wrong_count += 1
        ######
        # 对增强文本进行处理
        if ChatGPT_answers != "" and type(ChatGPT_answers)==list:
            augmented_sents = ChatGPT_answers[0].split('\n')
            n_augmented_sents = []
            for sent in augmented_sents:
                for extra_token in extra_tokens:
                    if extra_token in sent:
                        sent = sent.replace(extra_token, '')
                n_augmented_sents.append(sent)
            # 这个时候的n_augmented_sents就是augmented_sents处理后得到的纯正的sentences列表
            for sent in n_augmented_sents:
                keyphrases_info = {}
                for phrase in label:
                    start_idx = sent.find(phrase)
                    collected_info = []
                    while start_idx != -1:
                        collected_info.append([start_idx, start_idx + len(phrase) - 1])
                        start_idx = sent.find(phrase, start_idx + len(phrase))
                    if len(collected_info) > 0:
                        keyphrases_info[phrase] = collected_info
                if len(list(keyphrases_info)) != 0:
                    json_row = {'text': sent, 'label':{'financial_entity': keyphrases_info}}
                    oup1.write(json.dumps(json_row, ensure_ascii=False) + '\n')
                    oup2.write(json.dumps(json_row, ensure_ascii=False) + '\n')
                else:
                    print('未识别到keyphrase, 跳过.')
                    wrong_count += 1

    print('未记录数', wrong_count)

    # 将所有数据都准备好后，打乱train_new.json写入train_new_random.json.
    with open('./Fintech-Key-Phrase-new/train_new.json', mode='rt', encoding='utf-8') as oup3, \
            open('./Fintech-Key-Phrase-new/train_new_random.json', mode='wt', encoding='utf-8') as oup4:
        lines = oup3.readlines()
        random.shuffle(lines)
        oup4.writelines(lines)
