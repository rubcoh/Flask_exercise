from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
import pandas as pd

app = Flask(__name__)


pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


@app.route('/')
def index():
    return """
    <h1>Welcome to the main page</h1>

    <p>To make a single prediction, please add to the URL: /predict?load_model=1&age=..&sex=...&cp=...
    </p>
    <p>Here is a quick example to copy past: http://127.0.0.1:5000/predict?load_model=1&age=10&sex=1&cp=1
    </p>
    <p>The load_model parameter can take any value and is optional. load it only for the first time.
    </p>
    <p>
    Here are the names of the features:
    </p>
    <p>
    age, sex, cp
    </p>
    <p>
    To make multiple predictions, please make a POST request with a json file containing your data.
    </p>
    """

@app.route("/predict")
def predict():
    load_model = request.args.get('load_model')

    if load_model:
        pkl_filename = "/Users/ruben/Desktop/pickle_model.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)

    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')

    X_test = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp]})

    pickle_predict = pickle_model.predict(X_test)

    return f"""
   {int(pickle_predict)}
    """


@app.route('/json_predict', methods=['POST'])
def json_predict():
    data = request.get_json(force=True)

    X_test = pd.DataFrame(data)

    pickle_predict = pickle_model.predict(X_test)

    return f"""{pickle_predict}"""


if __name__ == '__main__':
    app.run(port=2000, debug=True)