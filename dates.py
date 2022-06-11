from flask import Flask, render_template, request
import time
import datetime
import pandas as pd

# from wtforms.fields.html5 import DateField
# from wtforms.validators import DataRequired
# form wtforms import validators, Submit
# from Flask_WTF import WTForm

df = pd.read_csv("datahargapangan nasional.csv", usecols=["Tanggal", "Daging Ayam Ras Segar"])
df2 = df["Tanggal"][-15:]
df3 = df["Daging Ayam Ras Segar"][-15:]



app = Flask(__name__)

@app.route("/")
def index():
  with open("datahargapangan nasional.csv") as file:
    return render_template("sampel.html", csv=file)

@app.route("/a")
def main():
    return render_template("sampel.html", df2=df2, df3=df3)

@app.route("/date", methods=["GET", "POST"])
def tanggal():
    if request.method == 'POST':
        tgl = request.form['timestep']
        current_date_temp = datetime.datetime.strptime(tgl, "%Y-%m-%d").date()
        newdate = current_date_temp + datetime.timedelta(days=5)
        # date = tgl.Convert.ToDateTime()
        # date = datetime.strptime(tgl, '%Y-%m-%d'))
        # timee = time.strptime(str(tgl),"%Y-%m-%d")
        # subm = request.form['tgl']
        # print(timee)
        return render_template("submit.html", tgl=tgl, newdate=newdate)

    
if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)