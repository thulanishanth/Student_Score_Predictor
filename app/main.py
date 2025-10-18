from flask import Flask, render_template, request
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'linear_regression_model.pkl')
model = joblib.load(os.path.abspath(MODEL_PATH))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('Hours', 0))
        predicted_score = model.predict(np.array([[hours]]))
        return render_template('index.html', prediction=f"Predicted Score: {predicted_score[0]:.2f}")
    except Exception as e:
        print("Error:", e)
        return render_template('index.html', prediction="Please enter a valid input")    

if __name__ == '__main__':
    app.run(debug=True)
