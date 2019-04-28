from datetime import datetime
import flask
from flask import Flask, render_template, make_response
from flask import request
# from flask_cors import CORS

from SentimentalPredictor import SentimentalPredictor

# from
import json
import base64

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/getcomment', methods=['POST'])
def get_commecnt():
    print("started")
    strLine = request.json['input']

    print('Input Natural query : ', strLine)
    sentimentalPredictor = SentimentalPredictor()
    prediction = sentimentalPredictor.sentimental_predict(strLine)

    json_data = {}
    json_data['status'] = "success"
    json_data['prediction'] = prediction
    json_data['message'] = ''
    resp = make_response(json.dumps(json_data), 200)
    resp.headers['Content-type'] = 'application/json; charset=utf-8'
    return resp


if __name__ == '__main__':
    app.run(port=5000, debug=True)
