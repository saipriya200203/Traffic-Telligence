import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=["POST"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'day', 'month', 'year', 'hours', 'minutes', 'seconds']
    data = pd.DataFrame(features_values, columns=names)
    data_scaled = scale.transform(data)
    prediction = model.predict(data_scaled)
    text = "The Estimated Traffic Volume is: "
    return render_template("output.html", prediction_text=text + str(prediction[0]))

if __name__ == "__main__":
    app.run(port=5000, debug=True)
