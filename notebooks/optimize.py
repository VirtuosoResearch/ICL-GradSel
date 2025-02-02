import numpy as np
from sklearn.linear_model import LinearRegression

data = [
    {"Ui": 1.5, "Si": [0, 1]},  # Ui 和对应的 Si
    {"Ui": 2.0, "Si": [1, 2]},
    {"Ui": 3.0, "Si": [0, 2]},
]


num_features = 3 
X = np.zeros((len(data), num_features))
y = np.zeros(len(data))

for i, entry in enumerate(data):
    Si = entry["Si"]
    Ui = entry["Ui"]
    y[i] = Ui
    for j in Si:
        X[i, j] = 1 

model = LinearRegression()
model.fit(X, y)

print("Learned coefficients (O):", model.coef_)
print("Intercept (bias):", model.intercept_)

y_pred = model.predict(X)
print("Predicted values:", y_pred)
print("True values:", y)
