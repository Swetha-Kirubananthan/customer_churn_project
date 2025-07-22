from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List
from pymongo import MongoClient
from datetime import datetime

# ⏳ Load the trained model
model = joblib.load("models/logistic_model.pkl")

# ⚙️ MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["churn_db"]
collection = db["predictions"]

# 🚀 FastAPI app
app = FastAPI()

# 📦 Input schema (based on features used in model)
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int
    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # 🔁 Rename input columns to match model's expected feature names
        rename_map = {
            "Contract_One_year": "Contract_One year",
            "Contract_Two_year": "Contract_Two year",
            "DeviceProtection_No_internet_service": "DeviceProtection_No internet service",
            "OnlineSecurity_No_internet_service": "OnlineSecurity_No internet service",
            "OnlineBackup_No_internet_service": "OnlineBackup_No internet service",
            "TechSupport_No_internet_service": "TechSupport_No internet service",
            "StreamingTV_No_internet_service": "StreamingTV_No internet service",
            "StreamingMovies_No_internet_service": "StreamingMovies_No internet service",
            "MultipleLines_No_phone_service": "MultipleLines_No phone service",
            "InternetService_Fiber_optic": "InternetService_Fiber optic",
            "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check"
        }

        # Convert input to DataFrame and rename columns
        df = pd.DataFrame([data.dict()])
        df.rename(columns=rename_map, inplace=True)

        # ✅ Make prediction
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df)[0][prediction]

        result = {
            "prediction": int(prediction),
            "confidence": round(float(confidence), 3)
        }

        # 💾 Log input and prediction to MongoDB
        collection.insert_one({
            "input": data.dict(),
            "renamed_input": df.to_dict(orient="records")[0],
            "output": result,
            "timestamp": datetime.now()
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))