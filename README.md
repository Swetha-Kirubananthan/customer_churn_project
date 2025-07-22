# 📉 Customer Churn Prediction API

This project predicts customer churn using a Logistic Regression model served via a FastAPI API. It includes real-time logging to MongoDB and can be tested using Postman.

---

## 🔧 Features

- 🎯 Predicts if a telecom customer is likely to churn
- 🚀 REST API built with FastAPI
- 📦 Trained Logistic Regression model
- 📊 Input/output logging in MongoDB
- 🧪 Easy testing with Postman or curl

---

## 🧠 Tech Stack

| Component        | Technology         |
|------------------|--------------------|
| Language         | Python             |
| ML               | Scikit-learn       |
| API              | FastAPI            |
| Database         | MongoDB (local)    |
| Data Processing  | Pandas             |
| Testing          | Postman, Curl      |
| Deployment       | Uvicorn            |
| Version Control  | Git, GitHub        |

---

<pre> ## 🗂️ Project Structure ``` customer_churn_project/ │ ├── data/ │ └── customer_data.csv # Raw dataset │ ├── models/ │ └── logistic_model.pkl # Trained model │ ├── notebooks/ │ └── churn_analysis.ipynb # EDA + training notebook │ ├── scripts/ │ ├── train_model.py # Script to train & save the model │ └── api.py # FastAPI prediction logic │ ├── main.py # Uvicorn runner for FastAPI ├── requirements.txt # All dependencies └── README.md # Project overview (this file) ``` </pre>