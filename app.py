from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Formula-based EMI
def calculate_emi(principal, annual_rate, years):
    monthly_rate = annual_rate / (12 * 100)
    months = years * 12
    emi = (principal * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return round(emi, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    emi = None
    ml_emi = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate'])
            tenure = float(request.form['tenure'])

            emi = calculate_emi(principal, rate, tenure)
            features = np.array([[principal, rate, tenure]])
            ml_emi = round(model.predict(features)[0], 2)
        except ValueError:
            emi = "Invalid input"
            ml_emi = None

    return render_template('index.html', emi=emi, ml_emi=ml_emi)

if __name__ == '__main__':
    app.run(debug=True)
