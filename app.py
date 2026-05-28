# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd   # ✅ keep imports together

# app = Flask(__name__)

# # Load model and encoders
# model = joblib.load('model.pkl')
# education_encoder = joblib.load('education_encoder.pkl')
# city_encoder = joblib.load('city_encoder.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():

#     experience = int(request.form['experience'])
#     age = int(request.form['age'])
#     education = request.form['education']
#     city = request.form['city']

#     education_encoded = education_encoder.transform([education])[0]
#     city_encoded = city_encoder.transform([city])[0]

#     # ✅ features now correctly inside the function
#     features = pd.DataFrame([{
#         'experience': experience,
#         'age': age,
#         'education': education_encoded,
#         'city': city_encoded
#     }])

#     prediction = model.predict(features)[0]

#     return render_template(
#         'index.html',
#         prediction_text=f"Predicted Salary: ₹ {round(prediction, 2)}"
#     )

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load model
model = joblib.load('models/model.pkl')
education_encoder = joblib.load('models/education_encoder.pkl')
city_encoder = joblib.load('models/city_encoder.pkl')

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Dashboard
@app.route('/dashboard')
def dashboard():

    df = pd.read_csv('data/employee_data.csv')

    plt.figure(figsize=(8,5))

    sns.histplot(df['salary'], kde=True)

    chart_path = 'static/charts/salary_distribution.png'

    plt.savefig(chart_path)

    plt.close()

    return render_template(
        'dashboard.html',
        chart=chart_path
    )

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    experience = int(request.form['experience'])
    age = int(request.form['age'])
    education = request.form['education']
    city = request.form['city']

    education_encoded = education_encoder.transform([education])[0]
    city_encoded = city_encoder.transform([city])[0]

    features = pd.DataFrame([{
        'experience': experience,
        'age': age,
        'education': education_encoded,
        'city': city_encoded
    }])

    prediction = model.predict(features)[0]

    return render_template(
        'index.html',
        prediction_text=f'Estimated Salary: ₹ {round(prediction, 2)}'
    )

if __name__ == "__main__":
    app.run(debug=True)