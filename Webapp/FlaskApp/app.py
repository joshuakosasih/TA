from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from postagger.Main import POSTagger
from nertagger.Main_Char import NERTagger as NerC
from nertagger.Main import NERTagger as Ner
import json

app = Flask(__name__)

data = {}

pt = 0
nt = 0
ntc = 0


def combineJSON(jstr, jstrc):
    arr_ori = json.loads(jstr)
    arr = arr_ori['labels']
    arrc = json.loads(jstrc)
    arrc = arrc['labels']
    for i, token in enumerate(arr):
        if token == '?':
            arr[i] = arrc[i]
    arr_ori['labels'] = arr
    return json.dumps(arr_ori)


@app.route("/")
def main():
    return render_template('index.html')


@app.route('/pos', methods=['GET'])
def posG():
    global pt
    if pt == 0:
        pt = POSTagger()
    return render_template('pos.html')


@app.route('/pos', methods=['POST'])
def posP():
    in_x = request.form['in_x']
    global pt
    json_data = pt.predict(in_x)
    print 'json data', json_data
    return render_template('pos.html', in_val=in_x, jsondata=json_data)


@app.route('/ner', methods=['GET'])
def nerG():
    global nt
    global ntc
    if nt == 0:
        nt = Ner()
    if ntc == 0:
        ntc = NerC()
    return render_template('ner.html')


@app.route('/ner', methods=['POST'])
def nerP():
    in_x = request.form['in_x']
    global nt
    json_data = nt.predict(in_x)
    print 'json data', json_data
    global ntc
    json_data_c = ntc.predict(in_x)
    print 'json data char', json_data_c
    json_data = combineJSON(json_data, json_data_c)
    return render_template('ner.html', in_val=in_x, jsondata=json_data)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
