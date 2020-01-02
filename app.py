import os
from flask import render_template, Flask, jsonify, request, send_from_directory
from cnn_predict import compare
from compair import compair


app = Flask(__name__)


@app.route("/")
def demo_page():
    return render_template('demo.html')


@app.route("/compare", methods=['POST'])
def compare_news():
    sen1 = request.form['sen1'].replace('@', '').replace('\r', '').replace('\n', ' ')
    sen2 = request.form['sen2'].replace('@', '').replace('\r', '').replace('\n', ' ')
    # print(sen1)
    # print(sen2)
    # print(compare(sen1, sen2))
    if len(sen1.split()) <= 300 or len(sen2.split()) <= 300:
        score = 5 * compair(sen1, sen2)
    else:
        score = compare(sen1, sen2)
    # print(score)
    return jsonify({'similarity': str(score)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)