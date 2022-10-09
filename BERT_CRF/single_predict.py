# -*- coding:utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__))
from transformers import BertTokenizer
import config
import numpy as np
import torch
from model import BertNER


def information_retrieval_crf(sents, is_roberta=False):
    device = torch.device("cuda:0")
    id2label = config.id2label
    if is_roberta:
        model_dir = r'C:\Users\Super-IdoI\Desktop\dataset-ecir\Fintech-Key-Phrase\BERT_CRF\experiments\financial\roberta_crf'  #  './experiments/financial/roberta_crf/'
    else:
        model_dir = r'C:\Users\Super-IdoI\Desktop\dataset-ecir\Fintech-Key-Phrase\BERT_CRF\experiments\financial\bert_crf'  # ./experiments/financial/bert_crf/'
    # "financial_entity": {"控制": [[10, 11]]}
    tokenizer = BertTokenizer.from_pretrained(config.roberta_model, do_lower_case=True)
    words = [list(text) for text in sents]
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
        batch_token_starts[idx][[idx for idx in token_start_idxs[idx] if idx < max_sent_len]]=1
    batch_data = torch.tensor(batch_data, dtype=torch.long).to(device)
    batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long).to(device)
    print("--------Load model from {}--------".format(model_dir))
    model = BertNER.from_pretrained(model_dir)
    print("--------Move model to {}--------".format(config.device))
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        batch_masks = batch_data.gt(0)
        label_masks = batch_masks[:,1:]
        batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]
        batch_output = model.crf.decode(batch_output, mask=label_masks)
        pred_tags = [[id2label.get(idx) for idx in indices] for indices in batch_output]
    results = []
    for sent, pred in zip(sents, pred_tags):
        entities = []
        entity = ''
        for word, tag in zip(sent,pred):
            if tag != 'O':
                entity += word
            else:
                if entity != '':
                    entities.append(entity)
                    entity = ''
        if entity != '':
            entities.append(entity)
        results.append({'sent':sent, 'pred_tags':pred,'phrases':entities})
    return results

if __name__ == '__main__':
    texts = ["报告期内，公司积极借鉴互联网思维，依托网络平台和工具，变革房地产销售模式，在部分城市发起全民经纪人等营销创新，主动整合渠道资源，取得较好成效。向城市配套服务商转型“和城市同步发展”是公司的一贯策略。",
        "争取在四氟系列产品、吡啶类含氟化学品和三氟甲基系列化学品进行突破",
             "发展彩色网络激光打印机、扫描仪、投影仪等产品",
             '滤波器等领域用R12K宽频、高BS、高居里温度材料（R12KB）材料开发、DM4550高性能永磁铁氧体材料”']
    res = information_retrieval_crf(texts, True)
    for prediction in res:
        print(f"{prediction['sent']}\n=> 标签:\n{prediction['pred_tags']}\n=> 包含实体:{prediction['phrases']}\n\n")
