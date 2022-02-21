from flask import Flask, render_template, request
import model



app = Flask(__name__)

@app.route("/")
def main():
    return render_template("dagingayam.html")

@app.route("/sub", methods=["POST"])
def submit():
    # HTML --> .PY
    if request.method == "POST":
        name=request.form["username"]

    # .PY --> HTML
    return render_template("submit.html", n=name)



if __name__ == '__main__':
    app.debug = True 
    app.run() 
    app.run(debug=True)