# importing the necessary dependencies
from flask import Flask, render_template, request,send_file,jsonify
from flask_cors import CORS,cross_origin

import pandas as pd
#import seaborn as sns
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import os
#sns.set()
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies=float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            Diabetes_Pedigree_Function = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            filename = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

            #loading Scaler pickle file
            scaler = pickle.load(open('standardScalar.sav', 'rb'))

            # predictions using the loaded model file and scalar file
            prediction = loaded_model.predict(scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi, Diabetes_Pedigree_Function, Age]]))

            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction==1:
                prediction = 'You are Diabetic!'
                return render_template('results.html', prediction=prediction)
            else:
                prediction = 'You are not Diabetic!'
                return render_template('results.html', prediction=prediction)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    #to run locally
    app.run(host='127.0.0.1', port=5000, debug=True)

    #to run on cloud
	#app.run(debug=True) # running the app