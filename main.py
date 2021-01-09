# importing the necessary dependencies
from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS, cross_origin

import pandas as pd

import pickle

# initialize the flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])  # Route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # Route to show the predictions in UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  Get the inputs entered by user
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            bmi = float(request.form['bmi'])
            Diabetes_Pedigree_Function = float(request.form['Diabetes_Pedigree_Function'])
            Age = float(request.form['Age'])

            filename = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

            # Load Standard Scalar pickle file
            scalar = pickle.load(open('standardScalar.sav', 'rb'))

            # Predictions using the loaded model file and scalar file
            prediction = loaded_model.predict(scalar.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                                 Insulin, bmi, Diabetes_Pedigree_Function, Age]]))

            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction == 1:
                prediction = 'You Are A Diabetic Patient.'
                return render_template('diabetes.html', prediction=prediction)
            else:
                prediction = 'You Are Not A Diabetic Patient.'
                return render_template('no_diabetes.html', prediction=prediction)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong. Please check the code!!'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # To run in local
    app.run(host='127.0.0.1', port=1229, debug=True)

    # To run in cloud
    # app.run(debug=True) # running the app
