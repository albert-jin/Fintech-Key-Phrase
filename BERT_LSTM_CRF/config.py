import os
import torch
torch.backends.cudnn.enabled = False

# data_dir = os.getcwd() + '/data/dataset_final/'
data_dir = '../Fintech-Key-Phrase/dataset_final/'

train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
# bert_model = '../pretrained_bert_models/bert-base-chinese/'
# roberta_model = '../pretrained_bert_models/chinese_roberta_wwm_large_ext/'
bert_model = r'C:\Users\Super-IdoI\Desktop\dataset-ecir\Fintech-Key-Phrase\pretrained_bert_models\bert-base-chinese' # './pretrained_bert_models/bert-base-chinese/'
roberta_model =  r'C:\Users\Super-IdoI\Desktop\dataset-ecir\Fintech-Key-Phrase\pretrained_bert_models\chinese_roberta_wwm_large_ext'  # '../pretrained_bert_models/chinese_roberta_wwm_large_ext/'
# roberta_model = '../pretrained_bert_models/bert-base-chinese/'

model_dir = os.getcwd() + '/experiments/financial/'
log_dir = model_dir + 'train_financial.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.05

# 是否加载训练好的NER模型
load_before = True  # False True

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 6
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['financial_entity']
label2id = {"O": 0,"B-financial_entity": 1,"I-financial_entity": 2, "S-financial_entity": 3}

id2label = {_id: _label for _label, _id in list(label2id.items())}
