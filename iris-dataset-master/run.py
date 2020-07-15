from flask import Flask, request, redirect, render_template, flash
from flask_restful import Api, Resource
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(i) for i in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    if output == 0:
        predicted_class = 'Iris-setosa'
    elif output == 1:
        predicted_class = 'Iris-versicolor'
    else:
        predicted_class = 'Iris-virginica'
    print(predicted_class)

    return render_template('index.html', predicted_class="Flower is {}.".format(predicted_class))


if __name__ == '__main__':
    app.run(debug=True)
