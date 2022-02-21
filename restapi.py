import json
from flask import Flask, render_template, redirect, url_for, request, jsonify
import numpy as np
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

# {"harga":[[0.41004184],[0.42259414],[0.41422594],[0.35146444],[0.33054393],[0.32635983],[0.34728033]]}




if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)