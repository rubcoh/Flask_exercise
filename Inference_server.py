from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

@app.route('/')
def main_page():
    return """Welcome to the main page"""


@app.route('/json_predict', methods=['POST'])
def json_predict():
    data = request.get_json(force=True)

    X_test = pd.DataFrame(data)

    pickle_predict = pickle_model.predict(X_test)

    return f"""{pickle_predict}"""


if __name__ == '__main__':
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))