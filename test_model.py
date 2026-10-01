import pickle
import pandas as pd

# Load saved model
with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example new customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 7,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0
}])

# Predict churn probability
churn_probability = model.predict_proba(customer)[0][1]

# Our selected threshold
threshold = 0.35

prediction = 1 if churn_probability >= threshold else 0

print(f"Churn Probability: {churn_probability:.2%}")

if prediction == 1:
    print("Prediction: Customer likely to CHURN")
else:
    print("Prediction: Customer likely to STAY")

