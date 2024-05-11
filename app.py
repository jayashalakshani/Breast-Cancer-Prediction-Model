import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('your_model.h5')
scaler = StandardScaler()

# Read the data_train CSV file
data_train = pd.read_csv('data_train.csv')

# Fit the scaler with your training data
scaler.fit(data_train.drop(columns='diagnosis')) 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(final_features)
        prediction = (prediction > 0.5)  # Convert probabilities to binary predictions
        if prediction == 0:
            result = 'Benign'
        else:
            result = 'Malignant'
        return render_template('result.html', prediction_text='The predicted diagnosis is {}'.format(result))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    features = np.array(list(data.values())).reshape(1, -1)
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)
    prediction = (prediction > 0.5)  # Convert probabilities to binary predictions
    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
