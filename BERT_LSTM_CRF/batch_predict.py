from transformers import BertTokenizer
import config
import numpy as np
import torch
from model import BertNER
import os
from docx import Document
import csv
from tqdm import tqdm

device = torch.device("cuda:0")
id2label = config.id2label
model_dir = './experiments/financial/'


def predict_from_docx_batches(input_dir, output_dir=None,  output_txt=True, output_csv=True):
    """
    读取input_dir下所有docx文件的内容，预测其中的金融相关实体，并将结果写入指定文件夹中
    :param input_dir: 指定需要预测一批docx文件的目录位置
    :param output_dir: 指定输出文件的目录位置，如不指定，则默认在 input_dir 同级文件夹下的“predict_results” 文件夹下
    :param output_txt: True：输出txt格式结果 False：不输出
    :param output_csv: True：输出csv格式结果 False：不输出
    :return: None
    """
    if not os.path.exists(input_dir):
        print(f'没有{input_dir}文件夹.')
        exit(-1)
    if not output_dir:
        father_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(father_dir, 'predict_results')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.abspath(output_dir)
    csv_filepath_o = os.path.join(output_dir, '预测实体集合.csv') if output_csv else ''
    csv_filepath = os.path.join(output_dir, '文档-实体集合.csv') if output_csv else ''
    print("""
    参数: 
    输入文档目录：{0}
    输出文档目录：{1}
    预计输出文档包含：
    {2}；
    {3}；
    {4}；
    """.format(os.path.abspath(input_dir), output_dir, csv_filepath_o, csv_filepath, '\"docx同名文件.txt\"' if output_txt else ''))
    print("--------Load vocab.txt from {}--------".format(config.roberta_model))
    tokenizer = BertTokenizer.from_pretrained(config.roberta_model, do_lower_case=True)
    print("--------Load model from {}--------".format(model_dir))
    model = BertNER.from_pretrained(model_dir)
    print("--------Move model to {}--------".format(config.device))
    model.to(config.device)
    print("-------- model evaluating... --------")
    model.eval()
    # model = 0
    # tokenizer = 0
    print(f"文档集合存储文件夹:{input_dir}.")
    if output_csv:
        # '文档-实体集合.csv'
        headers = ['filename', 'entities']
        values = []
        # '预测实体集合.csv'
        headers_o = ['entities']
        values_o = []
    for filename in os.listdir(input_dir):
        if filename not in ['601.docx', '602.docx', '603.docx']:
            continue
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.docx') and '$' not in filepath:  # 是待处理的原文档
            print(f"正在预测文档:{filename}...")
            file_content = ''
            docx_f = Document(filepath)
            for para in docx_f.paragraphs:
                file_content += para.text
            # 收集完文档的内容，开始对文档句子进行切分和batch预测
            sentences = []
            idx = 0  # 当前遍历文字下标
            while idx < len(file_content):
                idx_start = idx
                idx_end = idx + SENT_LEN
                sentence = file_content[idx_start:idx_end]
                if len(sentence) > OVERLAP_LEN:
                    sentences.append(sentence)
                idx = idx_end - OVERLAP_LEN

            entities = []  # 采集当篇文档中的所有金融实体
            for idx in tqdm(range(0, len(sentences), BATCH_SIZE)):  # 每批batch的配置
                batch_texts = sentences[idx:idx+BATCH_SIZE]
                if len(batch_texts) < 1:
                    break
                words = [list(text) for text in batch_texts]
                tokens = []
                for sentence in words:
                    token_ = []
                    for token in sentence:
                        token_.append(*tokenizer.tokenize(token))
                    tokens.append(token_)
                tokens = [['[CLS]'] + token for token in tokens]
                tokens = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
                token_start_idxs = [list(range(1, len(token))) for token in tokens]
                batch_len = len(tokens)
                max_sent_len = max([len(token) for token in tokens])
                batch_data = np.zeros((batch_len, max_sent_len))
                batch_token_starts = np.zeros((batch_len, max_sent_len))
                for idx, token in enumerate(tokens):
                    cur_len = len(token)
                    batch_data[idx][:cur_len] = token
                    batch_token_starts[idx][[idx for idx in token_start_idxs[idx] if idx < max_sent_len]] = 1
                batch_data = torch.tensor(batch_data, dtype=torch.long).to(device)
                batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long).to(device)
                with torch.no_grad():
                    batch_masks = batch_data.gt(0)
                    label_masks = batch_masks[:, 1:]
                    batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]
                    batch_output = model.crf.decode(batch_output, mask=label_masks)
                    pred_tags = [[id2label.get(idx) for idx in indices] for indices in batch_output]
                for sent, pred in zip(batch_texts, pred_tags):
                    entity = ''
                    for word, tag in zip(sent, pred):
                        if tag != 'O':
                            entity += word
                        else:
                            if entity != '':
                                if not len(entity) < 2 and not entity.isdigit():
                                    entities.append(entity)
                                entity = ''
                    if entity != '':
                        if not len(entity) < 2 and not entity.isdigit():
                            entities.append(entity)
                    # print(f'{sent}=> 包含实体:\n{entities}')
            if not IS_DUPLICATE:
                entities = list(set(entities))
            if output_csv:
                values.append((filename, str(entities)))
                values_o.extend([(entity,) for entity in entities])
            if output_txt:
                with open(os.path.join(output_dir,filename.replace(".docx",".txt")), encoding='utf-8', mode='wt') as fp:
                    fp.writelines('\n'.join(entities))
    if output_csv:
        with open(csv_filepath,'wt', encoding='utf-8',newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)
            writer.writerows(values)
        with open(csv_filepath_o,'wt', encoding='utf-8',newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(headers_o)
            writer.writerows(values_o)


# import csv
# headers = ['username','age','height']
# value = [
#     ('张三',18,180),
#     ('李四',19,175),
#     ('王五',20,170)
# ]
# with open("classroom.csv",'w',encoding='utf-8',newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(headers)
#     writer.writerows(value)

if __name__ == '__main__':
    INPUT_DIR = '../Fintech-Key-Phrase/annotated_docx'  # 源docx文件位置
    OUTPUT_DIR = '../Fintech-Key-Phrase/predict_results'  # 输出文件位置
    IS_OUTPUT_TXT = True  # 是否输出到txt
    IS_OUTPUT_CSV = True  # 是否输出到csv
    IS_DUPLICATE = False  # 是否对每篇文章的实体进行去重.
    BATCH_SIZE = 6  # 每次预测的句子数目
    OVERLAP_LEN = 8  # 相邻句子间重叠文字数目
    SENT_LEN = 64  # 每个句子的切分长度
    predict_from_docx_batches(INPUT_DIR, OUTPUT_DIR, IS_OUTPUT_TXT, IS_OUTPUT_CSV)





