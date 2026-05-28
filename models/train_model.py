import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/employee_data.csv")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Encoding
education_encoder = LabelEncoder()
city_encoder = LabelEncoder()

df['education'] = education_encoder.fit_transform(df['education'])
df['city'] = city_encoder.fit_transform(df['city'])

# Features and target
X = df[['experience', 'age', 'education', 'city']]
y = df['salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Advanced ML Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print(f"Model Accuracy: {score * 100:.2f}%")

# Save model
joblib.dump(model, 'models/model.pkl')
joblib.dump(education_encoder, 'models/education_encoder.pkl')
joblib.dump(city_encoder, 'models/city_encoder.pkl')

print("Model saved successfully!")