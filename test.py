from flask import Flask, render_template

app= Flask(__name__)

@app.route("/")
def home():
    data=[
        ("01/01/2021", 1597, 1432),
        ("02/01/2021", 1456, 1312),
        ("03/01/2021", 1908, 1801),
        ("04/01/2021", 896, 720),
        ("05/01/2021", 755, 600),
        ("06/01/2021", 453, 400),
        ("07/01/2021", 1110, 1547),
        ("08/01/2021", 1235, 1509),
        ("09/01/2021", 1478, 1830),
    ]

    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    values2= [row[2] for row in data]

    return render_template("sampel.html", labels=labels, values=values, values2=values2)

if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)