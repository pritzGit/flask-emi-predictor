import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

np.random.seed(42)
principal = np.random.randint(100000, 1000000, 100)
rate = np.random.uniform(5.0, 15.0, 100)
tenure = np.random.randint(1, 30, 100)

def calculate_emi(p, r, t):
    r = r / (12 * 100)
    n = t * 12
    emi = (p * r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

data = pd.DataFrame({
    'principal': principal,
    'rate': rate,
    'tenure': tenure
})

data['emi'] = calculate_emi(data['principal'], data['rate'], data['tenure'])

X = data[['principal', 'rate', 'tenure']]
y = data['emi']
model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
