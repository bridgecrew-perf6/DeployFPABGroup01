from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('BankModelDeploy.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def man():
    return render_template('index.html')

@app.route('/prediksi', methods=['POST'])
def ambil():
    age = int(request.form['age1'])
    job = int(request.form['job1'])
    marital = int(request.form['marital1'])
    education = int(request.form['education1'])
    balance = int(request.form['balance1'])
    housing = int(request.form['housing1'])
    loan = int(request.form['loan1'])
    arr= np.array([[age, job, marital, education, balance, housing, loan]])

    pred=model.predict(arr)
    return render_template('hasilprediksi.html', data=pred)

if __name__ == '__main__':
    app.run(debug=True)