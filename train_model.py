import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv("parkinsons.data")


# Separate input and output
X = data.drop(["name", "status"], axis=1)
y = data["status"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
result = model.predict(X_test)

accuracy = accuracy_score(y_test, result)


print("------------------------------")
print("Parkinson Detector Training")
print("------------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")


# Save model
joblib.dump(model, "parkinson_model.pkl")


print("Model saved successfully!")