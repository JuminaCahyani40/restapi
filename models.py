import pandas as pd
import numpy as np
# import csv
from tensorflow import keras
from tensorflow.keras import layers
# import math
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error

def prediksi_harga(tgl):
    df = pd.read_csv('data2020.csv', usecols=["Tanggal", "Daging Ayam"])
    # print(df)

    minm=min(df['Daging Ayam'])
    maxm=max(df['Daging Ayam'])
    df1 = df['Daging Ayam'].apply(lambda x:(x-minm)/(maxm-minm))
    # print(df1.shape)

    df2 = np.array(df1).reshape(-1,1)
    # print(df2,shape)

    no=len(df2)
    # print(len(x))
    timestep=7
    predictday=5
    exsam=[]
    yesam=[]
    for i in range(timestep, no-predictday):
        xsam=df2[i-timestep:i]
        ysam=df2[i:i+predictday]
        exsam.append(xsam)
        yesam.append(ysam)

    xdata=np.array(exsam)
    xdata=xdata.reshape(xdata.shape[0],xdata.shape[1],1)
    # print(xdata.shape)
    # print(xdata)


    ydata=np.array(yesam)
    ydata=ydata.reshape(ydata.shape[0],ydata.shape[1])
    # print(ydata.shape)
    # print(ydata)


    xdata_shape=xdata.shape[0]
    xdata_train=int(xdata_shape*0.70)

    ydata_shape=ydata.shape[0]
    ydata_train=int(ydata_shape*0.70)

    xtrain=xdata[:xdata_train]
    ytrain=ydata[:ydata_train]
    xtest=xdata[xdata_train:]
    ytest=ydata[ydata_train:]

    # print(xtrain.shape)
    # print(ytrain.shape)
    # print(xtest.shape)
    # print(ytest.shape)


    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model = keras.Sequential()
    model.add(layers.LSTM(50, input_shape=(7,1), return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    # model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(5))
    model.compile(loss='mean_squared_error', optimizer=optimizer)


    model.fit(xtrain, ytrain, batch_size=5, epochs=100)

    return model.predict(xtest)[0]

# prediksi_harga(harga)

# import h5py
# saving = h5py.File("models.hdf5", "w")

# loading = h5py.File('models.hdf5', 'r')



