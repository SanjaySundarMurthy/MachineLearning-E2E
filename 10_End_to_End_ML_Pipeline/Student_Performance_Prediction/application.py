import os,sys
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
from src.exception import customExcpetion
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import customData,predict_pipeline
application=Flask(__name__)

app=application

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_Data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            data=customData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("race_ethicnity"),
                parental_level_of_education=request.form.get(""),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=request.form.get("reading_score"),
                writing_score=request.form.get("writing_score")
            )
            pred_df=data.get_data_as_frame()
            
            print(pred_df)
            
            predict_pipe=predict_pipeline()
            results=predict_pipe.predict(pred_df)
            return render_template("home.html",results=results[0])
            
            
        except Exception as e:
            return customExcpetion(e,sys)


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)