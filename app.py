#### app.py ####

from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import statsmodels.api as sm    
from statsmodels.formula.api import ols

app = Flask(__name__)

# Training data
df = pd.read_excel("engagement_scores.xlsx")
# X = np.array([[0], [1], [2], [3], [4]])
# y = np.array([0, 2, 4, 6, 8])
# model = LinearRegression().fit(X, y)
Y = df['Engagement Score (Y^obs)']
X = df['Sustainability Spending (X)']
W = df['Treatment (W)']

# Create design matrix
X_design = sm.add_constant(pd.DataFrame({'W': W, 'X': X})) 

model = sm.OLS(Y, X_design).fit()

@app.route("/predict")
def predict():
    # x = float(request.args.get("x", 0))
    # y_pred = model.predict([[x]])[0]
    w = float(request.args.get("w", 0))  # Treatment indicator
    x = float(request.args.get("x", 0))  # Sustainability spending

    # Prepare input for prediction (must match training design matrix)
    input_data = pd.DataFrame({'const': [1], 'W': [w], 'X': [x]})

    # Make prediction
    y_pred = model.predict(input_data)[0]
    
    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input x: {x}, Input w: {w}\nPrediction: {y_pred}\n")
    
    return jsonify({"W": w, "X": x, "predicted_engagement_score": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



