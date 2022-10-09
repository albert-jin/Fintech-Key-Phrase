import json
def after_process(source_file, target_file):
    """ {"text": "着力提高各大电商平台阿米尼旗舰店销售能力和品牌宣传覆盖效果", "label": {"financial_entity": {"电商": [[6, 7]], "商平台": [[7, 9]], "电": [[6, 6]], "商": [[7, 7]]}}} """
    with open(source_file, mode='rt', encoding='utf-8') as inp, open(target_file, mode='wt', encoding='utf-8') as outp:
        lines = inp.readlines()
        for line in lines:
            obj = json.loads(line.strip())
            sent = obj['text']
            flags = [0]* len(sent)
            labels = obj['label']['financial_entity']
            for pos in labels.values():
                for i in pos:
                    for j in range(i[0],i[1]+1):
                        flags[j] = 1
            financial_entity = {}
            words = ''
            start_position = 0
            assert len(flags) == len(sent)
            for idx,(flag,word) in enumerate(zip(flags,sent)):
                if flag == 1:
                    if words == '':
                        start_position = idx
                    words += word
                    if idx == len(sent) - 1 and words != '':  # 如果句尾是短语且句尾的words不为空
                        if words in financial_entity:
                            financial_entity[words].append([start_position,idx])
                        else:
                            financial_entity[words] = [[start_position,idx]]
                        words = ''
                else:
                    if words != '':
                        if words in financial_entity:
                            financial_entity[words].append([start_position,idx-1])
                        else:
                            financial_entity[words] = [[start_position,idx-1]]
                        words = ''
            outp.write(json.dumps({"text":sent,"label":{"financial_entity":financial_entity}},ensure_ascii=False)+'\n')


if __name__ == '__main__':
    source_train = './dataset_final/train_pre.json'
    target_train = './dataset_final/train.json'
    after_process(source_train,target_train)

    source_test = './dataset_final/test_pre.json'
    target_test = './dataset_final/test.json'
    after_process(source_test,target_test)
