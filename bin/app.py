# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:57:13 2020

@author: Anaji
"""
from flask import Flask,render_template,url_for,request

from model import CreateModel

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
        Predict text spam or ham
    """
    if request.method == 'POST':
        message = request.form['message']
        message = [message]
        CreateModelObj = CreateModel()
        my_prediction = CreateModelObj.predictSpam(message)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)