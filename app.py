import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd 
import numpy as np

app=Flask(__name__)

##Load the model
regmodel=pickle.load(open("regmodel1.pkl","rb"))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=["post"])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[ float(x) for x in request.form.values()]   
    final_input=scaler.transform(np.array(data).reshape(1,-1)) 
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text=" The house price predicted is {} ".format(output))


if __name__=="__main__":
    app.run(debug=True)



