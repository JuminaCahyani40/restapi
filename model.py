# import csv
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# import statistics
# import math
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

df = pd.read_csv('data2020.csv', usecols=["Tanggal", "Daging Ayam"])

df2=df[["Daging Ayam"]]
scaler=MinMaxScaler()
data_fit=scaler.fit(df2)
data_scale=data_fit.transform(df2)
    
# print("Actual data", df2[0:5])
    # print("Scale Data", data_scale[0:5])
    # print(data_scale.shape)

no=len(data_scale)
timestep=7
predictday=5
exsam=[]
yesam=[]
for i in range(timestep, no-predictday):
    xsam=data_scale[i-timestep:i]
    ysam=data_scale[i:i+predictday]
    exsam.append(xsam)
    yesam.append(ysam)

xdata=np.array(exsam)
xdata=xdata.reshape(xdata.shape[0],xdata.shape[1],1)
# print(xdata.shape)


ydata=np.array(yesam)
ydata=ydata.reshape(ydata.shape[0],ydata.shape[1])
# print(ydata.shape)



xdata_shape=xdata.shape[0]
xdata_train=int(xdata_shape*0.80)

ydata_shape=ydata.shape[0]
ydata_train=int(ydata_shape*0.80)

xtrain=xdata[:xdata_train]
ytrain=ydata[:ydata_train]
xtest=xdata[xdata_train:]
ytest=ydata[ydata_train:]

# print(xtrain.shape)
# print(ytrain.shape)
# print(xtest.shape)
# print(ytest.shape)

# for inp, out in zip(xtrain[0:2], ytrain[0:2]):
#     print(inp)
#     print('====>')
#     print(out)
#     print('#'*20)


optimizer = keras.optimizers.Adam(learning_rate=0.01)
# optimizer = keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
#     name='RMSprop')

model = keras.Sequential()
model.add(layers.LSTM(50, input_shape=(7,1), return_sequences=True))
model.add(layers.LSTM(50, return_sequences=False))
# model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(5))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(xtrain, ytrain, batch_size=5, epochs=100)



import joblib

joblib.dump(data_fit, "scale.joblib")
model.save("model_pred.h5")

#  model.data_fit.inverse_transform(prediksi)

# return hasil_prediksi

# model.summary()
# model=get_model()

# model.save("my_model")

     
# train_prediksi = model.predict(xtrain)
# test_prediksi = model.predict(xtest)

# r2_score(ytrain,train_prediksi, multioutput='variance_weighted')

# r2_score(ytest,test_prediksi, multioutput='variance_weighted')

# math.sqrt(mean_squared_error(ytrain,train_prediksi))

# math.sqrt(mean_squared_error(ytest,test_prediksi))

# nilai_testPred = data_fit.inverse_transform(test_prediksi)
# print(nilai_testPred[0:5])

# nilai_testActual = data_fit.inverse_transform(ytest)
# print(nilai_testActual[0:5])

# X=nilai_testActual
# Y=nilai_testPred
# plt.scatter(X[:, 0], X[:, 1])
# plt.scatter(Y[:, 0], Y[:, 1])
# plt.show()

# data1=data_scale.shape[0]
# data2=data1-timestep
# data_7hariterakhir = data_scale[data2:]
# # print(data_7hariterakhir.shape)
# # data_7hariterakhir

# samples=1
# timest=data_7hariterakhir.shape[0]
# features=data_7hariterakhir.shape[1]

# data_direshape= data_7hariterakhir.reshape(samples, timest, features)
# data_direshape.shape

