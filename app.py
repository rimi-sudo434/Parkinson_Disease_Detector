import pandas as pd
import joblib


# Load trained model
model = joblib.load("parkinson_model.pkl")


# Load dataset
data = pd.read_csv("parkinsons.data")


# Remove unnecessary columns
X = data.drop(["name", "status"], axis=1)


print("==============================")
print(" Parkinson's Disease Detector ")
print("==============================")


# User input
patient = int(input("Enter patient number (0-194): "))


# Get patient data
sample = X.iloc[[patient]]


# Prediction
prediction = model.predict(sample)[0]


# Confidence
confidence = max(model.predict_proba(sample)[0]) * 100


print("\nPatient:", data.iloc[patient]["name"])

print("\nResult:")

if prediction == 1:
    print("⚠ Parkinson's Disease Detected")
else:
    print("✓ Healthy")


print("Confidence:", round(confidence, 2), "%")