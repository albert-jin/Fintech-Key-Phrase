# -*- coding:utf-8 -*-
import datetime

from flask import Flask, request
app = Flask(__name__)
from flask_cors import CORS
from flask import jsonify
from BERT_CRF.single_predict import information_retrieval_crf
from BERT_Softmax.single_predict import information_retrieval_linear
# from BERT_LSTM_CRF.single_predict import information_retrieval_lstm_crf
CORS(app)
from gevent import pywsgi
import re
is_debug = True
PORT = 8888

logs_txt = 'logs.txt'


def write_append(sent, logs_txt_ = logs_txt):
    with open(logs_txt_, encoding='utf-8',mode='at') as inp:
        inp.write(sent+'\n')


@app.route('/information_retrieval', methods=["POST"])
def information_retrieval():
    sents = ''
    try:
        if request.json:
            if request.json.get("sent") and request.json.get("model_type"):
                sents = request.json.get("sent")
                model_type = request.json.get("model_type")
            else:
                info_ = '未提供: 处理文本、模型类型.'
                write_append(str(datetime.datetime.now()) + info_)
                return jsonify({'result': 'failure', "info": info_})
        else:
            if request.form.get("sent") and request.form.get("model_type"):
                sents = request.form.get("sent")
                model_type = request.form.get("password")
            else:
                info_ = '未提供: 处理文本、模型类型.'
                write_append(str(datetime.datetime.now()) + info_)
                return jsonify({'status': 'failure', "info": info_})
        pattern = re.compile(r'\s+')
        sents = [re.sub(pattern,'', sent) for sent in sents]
        flag = False
        if 'roberta' in model_type:
            flag = True
        if 'lstm' in model_type:
            # res = information_retrieval_lstm_crf(sents, flag)
            res = information_retrieval_crf(sents, flag)  # 具体RoBerta-BiLSTM-CRF模型已经上传github
        elif 'crf' in model_type:
            res = information_retrieval_crf(sents, flag)
        else:
            res = information_retrieval_linear(sents, flag)
        write_append(str(datetime.datetime.now()) + ' 处理成功 ' + str(sents))
        return jsonify({'status': 'success', "results": res})
    except Exception as e:
        write_append(str(datetime.datetime.now()) + ' 处理失败 ' + str(sents) + str(e.args))
        return jsonify({'status': 'failure', "info": e.args})

if __name__ == '__main__':
    if is_debug:
        app.run('0.0.0.0', PORT, debug=is_debug)  # localhost写成0.0.0.0可在全网上访问
    else:
        server = pywsgi.WSGIServer(('0.0.0.0', PORT), app)
        server.serve_forever()

