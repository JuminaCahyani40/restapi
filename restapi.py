import json
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response, send_file
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import random
from io import StringIO
import base64
from tensorflow import keras
from tensorflow.keras import layers
import joblib

app = Flask(__name__)

# modeel = keras.models.load_model('my_model')
model=keras.models.load_model("model_pred.h5")
scaler=joblib.load("scale.joblib")

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/da", methods=["POST"])
def hargada():
    if request.method == 'POST':
        nama=request.form["subb"]
    return render_template('hello.html')


@app.route("/result", methods=["POST"])

def hasil():
    if request.method == 'POST':
        hello = request.form.getlist('data[]')
        hell=[float(i) for i in hello]
        hel=np.array(hell)
        sampel=1
        timestep=7
        feature=1
        dataset=hel.reshape(sampel, timestep, feature)
        prediksi=model.predict(dataset)
        hasilnya=scaler.inverse_transform(prediksi)
        subm = request.form['mit']
    return render_template('result.html', name=hasilnya)
   

# 34.800
# 34.800
# 34.800
# 34.800
# 34.800
# 34.900
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


@app.route("/plot", methods=['GET'])
def grafik():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    sb_x = ['10/9/2021', '11/9/2021', '12/9/2021', '13/9/2021', '14/9/2021']
    ypred = [32400, 32200, 32300, 32200, 32400]
    yact = [34800, 34800, 34800, 34800, 34800]
    axis.plot(sb_x, ypred)
    axis.plot(sb_x, yact)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')



if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)