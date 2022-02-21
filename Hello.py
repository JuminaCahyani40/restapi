from flask import Flask, redirect, render_template, url_for


#name adalah variabel turunan dari python, yang akan dijadikan sbg main (halaman utama)
#jika variabel nama akan di import dari file, maka variabel tsb diganti dg nama file yang akan diimport
#misalkan ada file test.py yang akan di import maka akan diganti menjadi app = Flask(test)
app = Flask(__name__)

@app.route('/')#route() adalah decorator yang akan menjelaskan ke Flask untuk menuju/menjalankan URL mana yang didefinisikan oleh method hello_world()
def hello_world(): #hello_world() adalah method. method ini bertanggung jawab untuk me return Hello World, yang menggunakan web browser dengan localhost:5000/(default)
    return 'Hello World!'

# @app.route('/admin')
# def hello_admin():
#     return 'hello admin'

# @app.route('/guest/<guest>')
# def hello_guest(guest):
#     return 'hello %s as guest' % guest

# @app.route('/user/<name>')
# def hello_user(name):
#     if name == 'admin':
#         return redirect(url_for('hello_admin')) #url_for untuk membangun URL secara dinamis. fungsi menerima argumen sesuai masing2 variabel URL
#     else:
#         return redirect(url_for('hello_guest', guest = name))


# @app.route('/hello/<name>') 
# def namanya(name): 
#     return 'Hello %s!' % name

# @app.route('/rev/<float:revNo>')
# def revisi(revNo):
#     return 'Revisian ke %f' % revNo

# @app.route('/go/<int:postID>')
# def nomor(postID):
#     return 'GO THE NEXT NUMBER IS %d!' % postID



# HTTP METHODS
# Protokol Http adalah dasar dari komunikasi data di world wide web. Metode pengambilan data yang berbeda dari URL yang ditentukan ditentukan dalam protokol ini.
# Method pada http protokol adalah
# GET : mengirim data dalam bentuk yang tidak terenkripsi ke server. metode yang paling umum
# HEAD : sama dengan GET, tapi tanpa response
# POST : digunakan untuk mengirim data dalam bentuk HTLM ke server. Data yang diterima dengan metode POST tidak di-cache oleh server.
# PUT : Mengganti semua representasi sumber daya target saat ini dengan konten yang diunggah.
# DELETE : Menghapus semua representasi saat ini dari sumber daya target yang diberikan oleh URL
# Secara default, rute Flask merespon permintaan GET. Namun, preferensi ini dapat diubah dengan memberikan argumen metode ke dekorator route().

# @app.route('/success/<name>')
# def success(name):
#     return 'welcome %s' % name

# @app.route('/login', methods = ['POST', 'GET'])
# def login():
#     if request.method == 'POST':
#         user = request.form['nm']
#         return redirect(url_for('success', name = user))
#     else:
#         user = request.args.get('nm')
#         return redirect(url_for('success', name = user))


# RENDER TEMPLATE
# File HTML dapat di render menggunakan fungsi render_template()
# maksud dr 'web templating system' merujuk pada design script HTML yang mana data variabelnya dapat dimasukkan secara dinamis
# flask menggunakan jinja2 template engine. 
# web templating system berisi syntax HTML yang disleingi placeholders untuk variabel dan expression python yang menggani nilai ketika template di render

# jinja2 menggunakan delimiters untuk mendapatkan return dari dokumen HTML
# {%...%} --> untuk pernyataan
# {{  }} --> untuk ekspresi yang akan me print output dari tmplate
# {#...#} --> untuk komen yang tidak disertakan di output template
# #...## --> untuk line statements


# @app.route('/hello/<user>')
# def hello_name(user):
#     return render_template('hello.html', name = user)

if __name__ == '__main__':
    app.debug = True #untuk mengaktifkan mode debug agar saat mendevelop, tinggal refresh jika ada perubahan kode.
    app.run() #method run() untuk menjalankan 
    app.run(debug=True) #mode debug diaktifkan dengan menyetel parameter True sebelum menjalankan/meneruskan ke metode run()




import csv
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import statistics
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


df = pd.read_csv('data2020.csv', usecols=["Tanggal", "Daging Ayam"])
df

df.plot(x='Tanggal', y='Daging Ayam')

df2=df[["Daging Ayam"]]
scaler=MinMaxScaler()
data_fit=scaler.fit(df2)
data_scale=data_fit.transform(df2)
print("Actual data", df2[0:5])
print("Scale Data", data_scale[0:5])
print(data_scale.shape)

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
print(xdata.shape)
# print(xdata)

ydata=np.array(yesam)
ydata=ydata.reshape(ydata.shape[0],ydata.shape[1])
print(ydata.shape)
# print(ydata)


xdata_shape=xdata.shape[0]
xdata_train=int(xdata_shape*0.80)

ydata_shape=ydata.shape[0]
ydata_train=int(ydata_shape*0.80)

xtrain=xdata[:xdata_train]
ytrain=ydata[:ydata_train]
xtest=xdata[xdata_train:]
ytest=ydata[ydata_train:]

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

for inp, out in zip(xtrain[0:2], ytrain[0:2]):
    print(inp)
    print('====>')
    print(out)
    print('#'*20)


optimizer = keras.optimizers.Adam(learning_rate=0.01)
# optimizer = keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
#     name='RMSprop')

model = keras.Sequential()
model.add(layers.LSTM(50, input_shape=(7,1), return_sequences=True))
model.add(layers.LSTM(50, return_sequences=False))
# model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(5))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])


model.summary()

model.fit(xtrain, ytrain, batch_size=5, epochs=100)

train_prediksi = model.predict(xtrain)
test_prediksi = model.predict(xtest)

r2_score(ytrain,train_prediksi, multioutput='variance_weighted')

r2_score(ytest,test_prediksi, multioutput='variance_weighted')

math.sqrt(mean_squared_error(ytrain,train_prediksi))

math.sqrt(mean_squared_error(ytest,test_prediksi))

nilai_testPred = data_fit.inverse_transform(test_prediksi)
print(nilai_testPred[0:5])

nilai_testActual = data_fit.inverse_transform(ytest)
print(nilai_testActual[0:5])

X=nilai_testActual
Y=nilai_testPred
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()

data1=data_scale.shape[0]
data2=data1-timestep
data_7hariterakhir = data_scale[data2:]
print(data_7hariterakhir.shape)
data_7hariterakhir

samples=1
timest=data_7hariterakhir.shape[0]
features=data_7hariterakhir.shape[1]

data_direshape= data_7hariterakhir.reshape(samples, timest, features)
data_direshape.shape

next5days=model.predict(data_direshape)
next5days

next5days=data_fit.inverse_transform(next5days)
next5days



















import csv
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import statistics
import math
from sklearn.metrics import r2_score
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv('datahargapangan nasional.csv', usecols=["Tanggal", "Daging Ayam"])
df

df.plot(x='Tanggal', y='Daging Ayam')

minm=min(df['Daging Ayam'])
maxm=max(df['Daging Ayam'])
df1 = df['Daging Ayam'].apply(lambda x:(x-minm)/(maxm-minm))
df1

mean = np.mean(df["Daging Ayam"])
print(mean)

median = np.median(df["Daging Ayam"])
print(median)

standardev = np.std(df["Daging Ayam"])
print(standardev)

def normal_distribution(x, mean, standardev):
    prob = 1/(2*(np.pi)*standardev) * np.exp(-0.5*((x-mean)/standardev)**2)
    return prob

pdf = normal_distribution(df["Daging Ayam"], mean, standardev)


plt.scatter(df["Daging Ayam"], pdf, color='red')
plt.xlabel('data points')
plt.ylabel('probability density')

print(min(df['Daging Ayam']))
print(df['Daging Ayam'])

df2 = np.array(df1).reshape(-1,1)
df2.shape
print(df2)

x=df2
no=len(df2)
# print(len(x))
timestep=7
predictday=5
exsam=[]
yesam=[]
for i in range(timestep, no-predictday):
    xsam=x[i-timestep:i]
    ysam=x[i:i+predictday]
    exsam.append(xsam)
    yesam.append(ysam)

xdata=np.array(exsam)
xdata=xdata.reshape(xdata.shape[0],xdata.shape[1],1)
print(xdata.shape)
# print(xdata)


ydata=np.array(yesam)
ydata=ydata.reshape(ydata.shape[0],ydata.shape[1])
print(ydata.shape)
# print(ydata)

xdata_shape=xdata.shape[0]
xdata_train=int(xdata_shape*0.70)

ydata_shape=ydata.shape[0]
ydata_train=int(ydata_shape*0.70)

xtrain=xdata[:xdata_train]
ytrain=ydata[:ydata_train]
xtest=xdata[xdata_train:]
ytest=ydata[ydata_train:]

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

for inp, out in zip(xtrain[0:2], ytrain[0:2]):
    print(inp)
    print('====>')
    print(out)
    print('#'*20)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
# optimizer = keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
#     name='RMSprop')

model = keras.Sequential()
model.add(layers.LSTM(50, input_shape=(7,1), return_sequences=True))
model.add(layers.LSTM(50, return_sequences=False))
# model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(5))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])

model.summary()

model.fit(xtrain, ytrain, batch_size=5, epochs=100)

train_prediksi = model.predict(xtrain)
test_prediksi = model.predict(xtest)

r2_score(ytrain,train_prediksi)
r2_score(ytest,test_prediksi)
math.sqrt(mean_squared_error(ytrain,train_prediksi))
math.sqrt(mean_squared_error(ytest,test_prediksi))

print(train_prediksi)

# print(len(train_prediksi))
# print(min(train_prJediksi))
# print(max(train_prediksi))
print(train_prediksi.shape)
# print(test_prediksi.shape)
# print(y_test.shape)

def denorm1(abc):
    p=[]
    for i in train_prediksi:
        u=np.array((i)*(maxm-minm)+minm)
        p.append([u])
    return(np.array(p))

trt=denorm1(train_prediksi)

print(trt.shape)
# print(trt)
print(test_prediksi.shape)
test1d = test_prediksi.flatten()
print(test1d.shape)


def denorm(dfg):
    c=[]
    for i in test_prediksi:
        b=np.array((i)*(maxm-minm)+minm)
        c.append([b])
    return(np.array(c))

tst=denorm(test_prediksi)
# print(tst)

print(tst.shape)

trt1 = np.array(trt)
tst1 = np.array(tst)
prediction = np.concatenate((trt1, tst1))
print(prediction)

a=df2.shape[0]
timestep=7
fix=a-timestep
# print(fix)
lst7hari = df2[fix:]
print(lst7hari.shape)
lst7hari

samples=1
timest=lst7hari.shape[0]
features=lst7hari.shape[1]


timest
features

lst7harii=lst7hari.reshape(samples, timest, features)
lst7harii.shape

next5days=model.predict(lst7harii)
next5days

# next5dayss=datasc.inverse_transform(next5days)
# next5dayss

def denorm1(abc):
    p=[]
    for i in next5days:
        u=np.array((i)*(maxm-minm)+minm)
        p.append([u])
    return(np.array(p))

predicsi=denorm1(next5days).flatten()

predicsi



