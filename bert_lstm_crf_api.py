# -*- coding:utf-8 -*-
from flask import Flask, request
app = Flask(__name__)
from flask_cors import CORS
from flask import jsonify
from BERT_LSTM_CRF.single_predict import information_retrieval_lstm_crf
CORS(app)
from gevent import pywsgi
is_debug = True
PORT = 8888


@app.route('/information_retrieval', methods=["POST"])
def information_retrieval():
    try:
        if request.json:
            if request.json.get("sent") and request.json.get("model_type"):
                sents = request.json.get("sent")
                model_type = request.json.get("model_type")
            else:
                return jsonify({'result': 'failure', "info": '未提供: 处理文本、模型类型.'})
        else:
            if request.form.get("sent") and request.form.get("model_type"):
                sents = request.form.get("sent")
                model_type = request.form.get("password")
            else:
                return jsonify({'status': 'failure', "info": '未提供: 处理文本、模型类型.'})
        flag = False
        if 'roberta' in model_type:
            flag = True
        res = information_retrieval_lstm_crf(sents, flag)
        return jsonify({'status': 'success', "results": res})
    except Exception as e:
        return jsonify({'status': 'failure', "info": e.args})

if __name__ == '__main__':
    if is_debug:
        app.run('0.0.0.0', PORT, debug=is_debug)  # localhost写成0.0.0.0可在全网上访问
    else:
        server = pywsgi.WSGIServer(('0.0.0.0', PORT), app)
        server.serve_forever()

