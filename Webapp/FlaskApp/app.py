from flask import Flask, render_template, request
from postagger.Main import POSTagger
import json

app = Flask(__name__)

data = {}

pt = POSTagger()

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/pos',methods=['GET'])
def posG():
    return render_template('pos.html')

@app.route('/pos',methods=['POST'])
def posP():
    in_x = request.form['in_x']
    global pt
    json_data = pt.predict(in_x)
    print 'json data', json_data
    return render_template('pos.html', in_val=in_x, jsondata=json_data)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
