from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load(r'D:\Shravani\climate-change-modeling\models\trained_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        year = int(request.form['year'])
        week_no = int(request.form['week_no'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        co2 = float(request.form['co2'])
        sea = float(request.form['sea_level'])
        precip = float(request.form['precipitation'])
        solar = float(request.form['solar'])

        features = [[year, week_no, latitude, longitude, co2, sea, precip, solar]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)  # Use original scaler for real app
        prediction = model.predict(scaled_features)[0]
        prediction = round(prediction, 3)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
