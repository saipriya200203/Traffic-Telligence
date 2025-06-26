HTTP:
python codes
To generate code for your Traffic Volume Estimation project, you can break it into four major components:


---

✅ 1. Data Preprocessing & Model Training

File: traffic_volume.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("traffic volume.csv")

# Preprocessing (example)
df = df.dropna()
le = LabelEncoder()
df['weather_main'] = le.fit_transform(df['weather_main'])

# Feature and target
X = df[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main']]
y = df['traffic_volume']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and encoder
pickle.dump(model, open('Flask/model.pkl', 'wb'))
pickle.dump(le, open('Flask/encoder.pkl', 'wb'))


---

✅ 2. Flask Web App

File: Flask/app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(_name_)
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features[-1] = encoder.transform([features[-1]])[0]  # encode 'weather_main'
    final_features = np.array([features])
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=f'Traffic Volume: {int(prediction[0])}')

if _name_ == "_main_":
    app.run(debug=True)


---

✅ 3. HTML Form

File: Flask/templates/index.html

<!DOCTYPE html>
<html>
<head><title>Traffic Volume Predictor</title></head>
<body>
    <h2>Enter the details</h2>
    <form method="POST" action="/predict">
        Temperature: <input type="text" name="temp"><br>
        Rain 1h: <input type="text" name="rain_1h"><br>
        Snow 1h: <input type="text" name="snow_1h"><br>
        Clouds: <input type="text" name="clouds_all"><br>
        Weather: <input type="text" name="weather_main"><br>
        <input type="submit" value="Predict">
    </form>
    <h3>{{ prediction_text }}</h3>
</body>
</html>


---

✅ 4. Requirements File

File: Requirements.txt

Flask
pandas
scikit-learn
numpy


---

✅ 5. IBM Watson Integration (Optional)

In IBM/traffic_volume_lbm_scoring_end_point.ipynb, you use IBM Watson ML API to deploy your model. Here's a basic template:

import requests
import json

API_KEY = "your_api_key"
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
)
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
payload_scoring = {
    "input_data": [{
        "fields": ["temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"],
        "values": [[294.26, 0.0, 0.0, 90, 1]]
    }]
}

response_scoring = requests.post(
    'your_ibm_endpoint_url',
    json=payload_scoring,
    headers=header
)

print("Prediction:", response_scoring.json())


---

Would you like me to generate a ZIP of all files together for download?
