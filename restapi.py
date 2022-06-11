from cProfile import label
import json
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response, send_file
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import io
import random
from math import sqrt
from io import StringIO
import base64
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import datetime
from sklearn.metrics import mean_squared_error
# import model

app = Flask(__name__)

# modeel = keras.models.load_model('my_model')
model=keras.models.load_model("daging_ayam/model_dgAyam.h5")
scaler=joblib.load("daging_ayam/scale.joblib")
r2score_training=joblib.load("daging_ayam/r2score_training.joblib")
r2score_testing=joblib.load("daging_ayam/r2score_testing.joblib")
rmse_training=joblib.load("daging_ayam/rmse_training.joblib")
rmse_testing=joblib.load("daging_ayam/rmse_testing.joblib")
df = pd.read_csv('datahargapangan nasional.csv')
df.to_csv("datahargapangan nasional.csv", index=None)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/da", methods=["POST"])
def hargada():
    if request.method == 'POST':
        nama=request.form["subb"]
        dataset=pd.read_csv("datahargapangan nasional.csv", usecols=["Tanggal", "Daging Ayam Ras Segar", "Beras Kualitas Bawah II", "Minyak Goreng", "Minyak Goreng Curah"])
        datast = dataset.tail(20)
        for i, row in datast.iterrows():
            x = (i, row["Daging Ayam Ras Segar"], row["Beras Kualitas Bawah II"], row["Minyak Goreng"], row["Minyak Goreng Curah"])
    return render_template('hello.html', datast=[datast.to_html()], titles=['dataset harga pangan'], x=x)

@app.route("/get_data", methods=["GET","POST"])
def hasil():
    if request.method == 'POST':
        tgl = request.form['timestep7']
        current_date = datetime.datetime.strptime(tgl, "%Y-%m-%d").date()
        newdate1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        newdate2 = (current_date + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        newdate3 = (current_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        newdate4 = (current_date + datetime.timedelta(days=4)).strftime("%Y-%m-%d")
        newdate5 = (current_date + datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        labels = [newdate1, newdate2, newdate3, newdate4, newdate5]
        get_data = request.form.getlist('data[]')
        list_getdata=[float(i) for i in get_data]
        toarray=np.array(list_getdata).reshape(-1,1)
        norm=scaler.transform(toarray)
        dataset=norm.reshape(1, 7, 1)
        prediksi=model.predict(dataset)
        hasilnya=scaler.inverse_transform(prediksi)
        hsl=np.array(hasilnya)
        hsl2=hsl.flatten()
        data_prediksi=hsl2.tolist()
        subm = request.form['mit']
        # labels=["10/9/2021", "11/9/2021","12/9/2021","13/9/2021","14/9/2021"]
        values=['34800', '34800', '34800', '34800', '34800']
        values2=[34800, 34800, 34800, 34800, 34800]
        rmse=sqrt(mean_squared_error(values2, data_prediksi))
        r2sc_tr=r2score_training
        r2sc_ts=r2score_testing
        rmse_tr=rmse_training
        rmse_ts=rmse_testing
    return render_template('dagingayam.html', hsl=hsl2, newdate1=newdate1, newdate2=newdate2, newdate3=newdate3, newdate4=newdate4, newdate5=newdate5, values2=data_prediksi, labels=labels, values=values, r2sc_tr=r2sc_tr, r2sc_ts=r2sc_ts, rmse_tr=rmse_tr, rmse_ts=rmse_ts, rmse=rmse)



if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)